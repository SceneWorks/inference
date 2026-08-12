//! FLUX.2-klein **reference-image edit** provider (sc-5487, epic 5480) — Kontext-style edit / identity
//! conditioning on FLUX.2-klein-9B off-Mac (Windows/CUDA), the candle sibling of the `mlx-gen-flux2`
//! edit variant (`flux2_klein_9b_edit`) and the **provider half** that unblocks the worker wiring.
//! FLUX.2-klein has no torch path (it is diffusers/MLX-only), so this lane retires the worker's
//! `edit_image` → torch deferral for `flux2_klein_9b`.
//!
//! **How it conditions (no transformer change):** each reference image is VAE-encoded into the packed,
//! bn-normalized transformer latent ([`Flux2Vae::encode_packed`]) and packed to tokens, then
//! concatenated AFTER the noised target tokens on the sequence axis — the joint image stream
//! `[target, ref0, ref1, …]`. The reference grid ids are offset at `t = 10 + 10·i` (the mlx fork's
//! per-reference temporal coordinate) so the 4-axis RoPE keeps the references positionally distinct
//! from the `t = 0` target grid. The existing [`Flux2Transformer::forward`] already accepts arbitrary
//! `img_ids`, so it runs the full joint sequence unchanged; the provider keeps the leading `target_seq`
//! velocity tokens and steps only the target. The reference tokens are clean and constant across the
//! denoise (re-concatenated each step, never noised).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — mirroring the SDXL edit /
//! IP-Adapter / InstantID / PuLID providers. Determinism is the candle-lane contract (sc-3673): the
//! seeded CPU init noise reuses [`pipeline::create_noise`]. Distilled klein runs CFG-free (guidance
//! 1.0); guidance > 1 adds a classifier-free negative pass (the same convention as txt2img). No
//! `strength`: FLUX.2 edit conditions via reference token concat (a full denoise from noise), not an
//! img2img noise blend.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    GenerationMemory, Image, MemoryProviderContract, MemoryRunContext, PidWeights, PreviewSink,
    Progress, Quant,
};
// `LatentDecoder` brings the `PidDecoder::decode` trait method into scope (sc-8044).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE, DEFAULT_STEPS, SIZE_MULTIPLE};
use crate::text_encoder::Flux2PromptEncoder;
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;
use crate::{pipeline, to_image, Pipeline, PID_BACKBONE};

/// Path to the FLUX.2 edit snapshot — just the diffusers snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`), the same snapshot the txt2img path loads. klein at
/// `black-forest-labs/FLUX.2-klein-9B` ([`Flux2Edit::load`]); dev at `black-forest-labs/FLUX.2-dev`
/// ([`Flux2Edit::load_dev`], sc-7460).
pub struct Flux2EditPaths {
    /// FLUX.2 diffusers snapshot dir (klein or dev).
    pub root: PathBuf,
}

/// One FLUX.2-klein edit request.
#[derive(Clone)]
pub struct Flux2EditRequest {
    pub prompt: String,
    /// Classifier-free negative prompt — used only when `guidance > 1` (distilled klein runs CFG-free).
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Guidance scale. 1.0 (klein default) = a single CFG-free forward; > 1.0 adds a negative pass.
    pub guidance: f32,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](Flux2Edit::with_pid), the final latent is decoded by the `flux2` PiD
    /// student (4× SR → 2K/4K) instead of the native FLUX.2 VAE. `false` (default) keeps the VAE decode.
    pub use_pid: bool,
    /// Per-step latent-preview sink (epic 16948, sc-16955) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which this provider cannot carry because it is worker-invoked by name rather than through a
    /// registered descriptor. Default is inert, and an inert sink makes a seeded render
    /// byte-identical to one with no preview at all.
    ///
    /// The frames project the **target** token stream only: the reference tokens are concatenated
    /// inside the predict closure and sliced back off, so they are never part of the running latent.
    pub preview: PreviewSink,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux2EditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS as usize,
            guidance: DEFAULT_GUIDANCE,
            seed: 0,
            use_pid: false,
            preview: PreviewSink::default(),
            cancel: CancelFlag::default(),
        }
    }
}

/// Loaded FLUX.2-klein edit model: the Qwen3 text encoder + the MMDiT + the VAE **with the encoder**
/// (the reference encode), plus the txt2img `Pipeline` handle (snapshot mmap + prompt encode + the
/// latent geometry/dtype). `generate` takes `&self` (no per-call mutation), so one load serves many
/// edits.
pub struct Flux2Edit {
    pipe: Pipeline,
    variant: Flux2Variant,
    te: Option<Flux2PromptEncoder>,
    /// Prompt tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt/branch.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: Option<Flux2Transformer>,
    vae: Option<Flux2Vae>,
    memory: GenerationMemory,
    loaded_quant: Option<Quant>,
    memory_contract: Option<MemoryProviderContract>,
    admitted_context: Option<MemoryRunContext>,
    lifecycle: std::sync::Mutex<()>,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// FLUX.2 edit composes the FLUX.2 VAE, so it loads the SAME `flux2` student ([`PID_BACKBONE`]) as the
    /// registered FLUX.2 provider.
    pid: Option<PidEngine>,
}

impl Flux2Edit {
    /// Load the **klein** edit backbone (dense) with the VAE encoder enabled (the reference encode);
    /// distilled — guidance 1.0 (CFG-free), > 1 adds a negative pass.
    pub fn load(paths: &Flux2EditPaths) -> Result<Self> {
        Self::load_variant(paths, Flux2Variant::Klein9b, None)
    }

    /// Load the **dev** edit backbone (sc-7460): the 32B flagship via the CPU-stage → quantize-onto-GPU
    /// loader (`quant` Q4/Q8 required in practice — the dense 32B does not fit the GPU), guidance-
    /// distilled (embedded scalar, no negative pass), text-only Mistral prompt + reference token concat.
    pub fn load_dev(paths: &Flux2EditPaths, quant: Option<Quant>) -> Result<Self> {
        Self::load_variant(paths, Flux2Variant::Dev, quant)
    }

    /// Load the dev edit lane with the exact request-scoped memory realization selected by the
    /// shared ladder. Staged loads retain only the lightweight snapshot handle and tokenizer;
    /// conditioning and heavy components are materialized in disjoint generation phases.
    pub fn load_dev_with_memory(
        paths: &Flux2EditPaths,
        quant: Option<Quant>,
        memory: GenerationMemory,
    ) -> Result<Self> {
        Self::load_variant_with_memory(paths, Flux2Variant::Dev, quant, memory)
    }

    /// Load the dev edit lane without a request-scoped ladder context while preserving the complete
    /// caller-authored [`LoadSpec`](candle_gen::gen_core::LoadSpec), including a validated text
    /// encoder substitution. Constrained memory still requires the context-bearing constructor.
    pub fn load_dev_with_memory_spec(
        paths: &Flux2EditPaths,
        quant: Option<Quant>,
        spec: &candle_gen::gen_core::LoadSpec,
        memory: GenerationMemory,
    ) -> Result<Self> {
        Self::load_variant_with_memory_spec(paths, Flux2Variant::Dev, quant, spec, memory)
    }

    /// Klein counterpart to [`Self::load_dev_with_memory_spec`].
    pub fn load_klein_with_memory_spec(
        paths: &Flux2EditPaths,
        spec: &candle_gen::gen_core::LoadSpec,
        memory: GenerationMemory,
    ) -> Result<Self> {
        Self::load_variant_with_memory_spec(paths, Flux2Variant::Klein9b, None, spec, memory)
    }

    /// Context-bearing ladder entry point used by bespoke worker routes. This preserves the same
    /// ABI/fingerprint/tier/route fail-closed boundary as the registered generator before reducing
    /// the admitted selection to its execution knobs.
    pub fn load_dev_with_memory_context(
        paths: &Flux2EditPaths,
        quant: Option<Quant>,
        spec: &candle_gen::gen_core::LoadSpec,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> Result<Self> {
        Self::load_with_memory_context(paths, Flux2Variant::Dev, quant, spec, context)
    }

    /// Klein counterpart to [`Self::load_dev_with_memory_context`]. All four worker route surfaces
    /// (edit, character, reference, and style) enter this one provider primitive; their evidence and
    /// catalog calibration remain distinct in SceneWorks.
    pub fn load_klein_with_memory_context(
        paths: &Flux2EditPaths,
        spec: &candle_gen::gen_core::LoadSpec,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> Result<Self> {
        Self::load_with_memory_context(paths, Flux2Variant::Klein9b, None, spec, context)
    }

    fn load_with_memory_context(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        quant: Option<Quant>,
        spec: &candle_gen::gen_core::LoadSpec,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> Result<Self> {
        validate_base_binding(paths, spec)?;
        validate_memory_load_spec(variant, spec)?;
        let loaded_quant = crate::memory_strategy::resolved_quant(spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let contract = crate::memory_strategy::contract_for_variant(variant, spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(&contract, context, loaded_quant)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        if quant.is_some() && quant != loaded_quant {
            return Err(CandleError::Msg(format!(
                "flux2 edit: requested {quant:?} but the admitted snapshot resolves to {loaded_quant:?}"
            )));
        }
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        let text_encoder_source = variant
            .encoder_contract()
            .source_for_load(spec, &paths.root)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        text_encoder_source
            .load_time_quant_bits(
                variant
                    .is_dev()
                    .then_some(loaded_quant)
                    .flatten()
                    .map(Quant::bits),
                variant.id(),
            )
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        Self::load_variant_bound(
            paths,
            variant,
            loaded_quant,
            memory,
            Some(contract),
            Some(context.clone()),
            text_encoder_source,
        )
    }

    /// Shared loader: the backbone for `variant` with the VAE encoder enabled (the reference encode).
    /// The dev quant path stages the TE + DiT dense in CPU RAM and quantizes each projection onto the
    /// GPU; klein (and dev on a fixture) loads dense on-device. f32 compute (parity-sensitive).
    fn load_variant(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Self::load_variant_with_memory(paths, variant, quant, GenerationMemory::default())
    }

    fn load_variant_with_memory(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        quant: Option<Quant>,
        memory: GenerationMemory,
    ) -> Result<Self> {
        let mut spec = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(paths.root.clone()),
        );
        spec.quantize = quant;
        Self::load_variant_with_memory_spec(paths, variant, quant, &spec, memory)
    }

    fn load_variant_with_memory_spec(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        quant: Option<Quant>,
        spec: &candle_gen::gen_core::LoadSpec,
        memory: GenerationMemory,
    ) -> Result<Self> {
        validate_memory_authority(memory, None, "flux2 edit")?;
        validate_base_binding(paths, spec)?;
        validate_memory_load_spec(variant, spec)?;
        let loaded_quant = crate::memory_strategy::resolved_quant(spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        if quant.is_some() && quant != loaded_quant {
            return Err(CandleError::Msg(format!(
                "flux2 edit: requested {quant:?} but the authored snapshot resolves to {loaded_quant:?}"
            )));
        }
        let text_encoder_source = variant
            .encoder_contract()
            .source_for_load(spec, &paths.root)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        text_encoder_source
            .load_time_quant_bits(
                variant
                    .is_dev()
                    .then_some(loaded_quant)
                    .flatten()
                    .map(Quant::bits),
                variant.id(),
            )
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        Self::load_variant_bound(
            paths,
            variant,
            loaded_quant,
            memory,
            None,
            None,
            text_encoder_source,
        )
    }

    fn load_variant_bound(
        paths: &Flux2EditPaths,
        variant: Flux2Variant,
        loaded_quant: Option<Quant>,
        memory: GenerationMemory,
        memory_contract: Option<MemoryProviderContract>,
        admitted_context: Option<MemoryRunContext>,
        text_encoder_source: candle_gen::gen_core::ValidatedEncoderSource,
    ) -> Result<Self> {
        validate_memory_authority(memory, admitted_context.as_ref(), "flux2 edit")?;
        let optimized =
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks;
        if optimized && !memory.stage_residency {
            return Err(CandleError::Msg(
                "flux2 edit: optimized memory rungs require staged residency".into(),
            ));
        }
        let device = candle_gen::default_device()?;
        // PiD (super-resolving decode) is wired only through the txt2img render path (epic 7840 /
        // sc-7853); the edit provider passes `None`.
        let pipe = Pipeline::load_with_text_encoder(
            variant,
            loaded_quant,
            &paths.root,
            text_encoder_source,
            &device,
            None,
        );
        // Packed MLX tier → build directly on the GPU from the packed parts (sc-9087, no ~105 GB dense
        // CPU staging); dense tier → the legacy CPU-stage → quantize-onto-GPU path. Shared TE+DiT loader
        // with txt2img / control (F-024, sc-9004). The VAE *with encoder* (the reference encode) is the
        // per-site addition.
        let (te, transformer, vae) = if memory.stage_residency {
            (None, None, None)
        } else {
            let (te, transformer) = pipe.load_te_and_dit()?;
            let vae = Flux2Vae::new_with_encoder(pipe.component_vb("vae")?)?;
            (Some(te), Some(transformer), Some(vae))
        };
        let tokenizer = pipe.build_tokenizer()?;
        Ok(Self {
            pipe,
            variant,
            te,
            tokenizer,
            transformer,
            vae,
            memory,
            loaded_quant,
            memory_contract,
            admitted_context,
            lifecycle: std::sync::Mutex::new(()),
            pid: None,
        })
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry FLUX.2 provider; edit composes the FLUX.2 VAE so it loads the **same**
    /// `PID_BACKBONE` (`flux2`) student. A `use_pid = true` request then decodes through it (4× SR)
    /// instead of the native VAE; without it, `use_pid` errors loudly. Call after [`load`](Self::load).
    pub fn with_pid(mut self, pid: &PidWeights) -> Result<Self> {
        self.pid = Some(PidEngine::from_spec(pid, PID_BACKBONE, &self.pipe.device)?);
        Ok(self)
    }

    /// Mint the per-generation PiD decoder when the request opted in (`use_pid`) and a student is loaded;
    /// `None` keeps the native VAE decode. Errors loudly if `use_pid` is set without a prior
    /// [`with_pid`](Self::with_pid). A clean-latent (σ=0) decoder bound to the prompt + seed; the request
    /// cancel threads in for a cancellable SR decode.
    fn pid_decoder_for(&self, req: &Flux2EditRequest) -> Result<Option<PidDecoder>> {
        // Route through the shared guarded seam (sc-11242 / F-091) so the SR decode is budgeted
        // (F-013 sc-9095) and spatially tiled (sc-10087). Clean-latent σ=0 decode, single image.
        candle_gen_pid::resolve_pid_decoder_for_fields(
            self.pid.as_ref(),
            req.use_pid,
            &req.prompt,
            1,
            req.width,
            req.height,
            &req.cancel,
            req.seed,
            "flux2 edit",
            0.0,
        )
    }

    /// Generate one edited image. `references` (≥ 1) condition the denoise via reference token concat;
    /// the worker pre-fits them to the render size, but this re-resizes defensively.
    pub fn generate(
        &self,
        req: &Flux2EditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        crate::run_bespoke_request(
            &self.lifecycle,
            || {
                ensure_ordinary_generate_allowed(self.admitted_context.as_ref(), "flux2 edit")?;
                validate_memory_authority(
                    self.memory,
                    self.admitted_context.as_ref(),
                    "flux2 edit",
                )?;
                self.generate_inner(req, references, on_progress)
            },
            || self.pipe.device.synchronize(),
        )
    }

    /// Execute the exact edit request admitted at load. The context identity and request geometry are
    /// rechecked immediately before execution; the request is serialized and the device is fenced on
    /// success, cancellation, and error.
    pub fn generate_with_memory_context(
        &self,
        context: &MemoryRunContext,
        req: &Flux2EditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        crate::run_bespoke_request(
            &self.lifecycle,
            || {
                let admitted = self.admitted_context.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "flux2 edit: model was not loaded with a memory context".to_owned(),
                    )
                })?;
                let contract = self.memory_contract.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "flux2 edit: admitted model lost its memory contract".to_owned(),
                    )
                })?;
                validate_admitted_context(admitted, context, "flux2 edit")?;
                validate_memory_request(context, req, references.len())?;
                crate::memory_strategy::validate_context(contract, context, self.loaded_quant)
                    .map_err(|error| CandleError::Msg(error.to_string()))?;
                self.generate_inner(req, references, on_progress)
            },
            || self.pipe.device.synchronize(),
        )
    }

    fn generate_inner(
        &self,
        req: &Flux2EditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        if references.is_empty() {
            return Err(CandleError::Msg(
                "flux2 edit: at least one reference image is required".into(),
            ));
        }
        validate_request(req)?;

        if self.memory.stage_residency && req.use_pid {
            return Err(CandleError::Msg(
                "flux2 edit: optimized native-VAE memory rungs do not support PiD".into(),
            ));
        }

        let device = &self.pipe.device;
        let cfg = &self.pipe.cfg;
        let guidance = req.guidance;
        // dev is guidance-distilled (embedded scalar, single forward); klein is distilled / true-CFG
        // (a classifier-free negative pass only when guidance > 1).
        let embedded_guidance = self.variant.uses_embedded_guidance();
        let cfg_on = !embedded_guidance && guidance > 1.0;

        // Prompt embeds are seed-independent: encode once. Negative only under klein CFG.
        let encode_prompt = |te: &Flux2PromptEncoder| -> Result<(Tensor, Option<Tensor>)> {
            let prompt = self.pipe.encode(te, &self.tokenizer, &req.prompt)?;
            let negative = if cfg_on {
                let neg = if req.negative.trim().is_empty() {
                    " "
                } else {
                    req.negative.as_str()
                };
                Some(self.pipe.encode(te, &self.tokenizer, neg)?)
            } else {
                None
            };
            Ok((prompt, negative))
        };
        let (prompt_embeds, negative) = if let Some(te) = self.te.as_ref() {
            encode_prompt(te)?
        } else {
            candle_gen::check_cancel(&req.cancel)?;
            let te = self.pipe.load_te_seq()?;
            let result = encode_prompt(&te);
            let result = candle_gen::synchronize_result(&self.pipe.device, result);
            drop(te);
            result?
        };

        // Reference conditioning: VAE-encode each ref → packed tokens [1, seq_ref, 128] + grid ids at
        // t = 10 + 10·i, all concatenated on the sequence axis. Clean + constant across the denoise.
        let (ref_tokens, ref_ids) = if let Some(vae) = self.vae.as_ref() {
            self.encode_references(vae, references, req.width, req.height)?
        } else {
            candle_gen::check_cancel(&req.cancel)?;
            let vae = Flux2Vae::new_with_encoder(self.pipe.component_vb("vae")?)?;
            let result = self.encode_references(&vae, references, req.width, req.height);
            let result = candle_gen::synchronize_result(&self.pipe.device, result);
            drop(vae);
            result?
        };

        // The staged heavy phase starts only after both conditioning owners have synchronized and
        // dropped. The base DiT may be a one-block window over host-backed weights; the decode VAE
        // stays on CPU for the bounded rung so it contributes no accelerator residency spike.
        let staged_transformer = if self.transformer.is_none() {
            candle_gen::check_cancel(&req.cancel)?;
            Some(
                self.pipe
                    .load_dit_seq_with_memory(self.memory.stream_transformer_blocks)?,
            )
        } else {
            None
        };
        let staged_vae = if self.vae.is_none() {
            let vae_device = if self.memory.tile_vae_decode {
                Device::Cpu
            } else {
                self.pipe.device.clone()
            };
            Some(Flux2Vae::new(
                self.pipe.component_vb_on("vae", &vae_device)?,
            )?)
        } else {
            None
        };
        let transformer = self
            .transformer
            .as_ref()
            .or(staged_transformer.as_ref())
            .expect("resident or staged transformer");
        let vae = self
            .vae
            .as_ref()
            .or(staged_vae.as_ref())
            .expect("resident or staged vae");

        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let target_seq = lat_h * lat_w;
        // The joint image-stream ids: the t=0 target grid followed by the reference grids.
        let mut img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        img_ids.extend_from_slice(&ref_ids);
        let txt_ids = pipeline::prepare_text_ids(cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123) — the same driver the txt2img path
        // uses. The bespoke edit request carries no per-generation sampler/scheduler knob, so this runs
        // the default (`None`) euler over the native empirical-mu schedule: the N1 no-op that reproduces
        // the legacy `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), req.steps);
        let native = pipeline::schedule(req.steps, req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);

        let latents = pipeline::create_noise(cfg, req.seed, req.width, req.height, device)?;
        // Per-step latent preview (epic 16948, sc-16955), bound to the same `(lat_h, lat_w)` the decode
        // tail below unpacks against. The sampler's running latent is the TARGET token grid alone —
        // `ref_tokens` is concatenated and sliced back off inside the predict closure — so the hook
        // structurally cannot project a reference image.
        let preview = crate::preview::hook(&req.preview, vae, lat_h, lat_w);
        let attention_budget = if self.memory.chunk_attention {
            self.memory
                .attention_chunk_size
                .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE) as u64
        } else {
            candle_gen::ATTN_SCORES_BUDGET as u64
        };
        let attention_plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            attention_budget,
            false,
        ));
        let attention_plan = if self.memory.chunk_attention {
            attention_plan.with_cancel(&req.cancel)
        } else {
            attention_plan
        };
        let transformer_window = self
            .memory
            .transformer_window_size
            .map(|value| value as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        // The driver does cancel + progress + the integrator step. The joint `[target, refs]` concat,
        // the transformer forward, the target-slice, and the guidance>1 CFG blend all live inside the
        // predict closure so a multi-eval solver re-runs them. FLUX.2 uses the Sigma convention but the
        // model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
        let latents = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            Some(&preview),
            |latents, sigma| -> Result<Tensor> {
                let ts = sigma * 1000.0;
                // Joint image stream [target, refs] — references re-concatenated with the current target.
                let hidden = Tensor::cat(&[latents, &ref_tokens], 1)?;
                if embedded_guidance {
                    // dev: a single forward feeding the embedded guidance scalar to the DiT.
                    return self.velocity(
                        transformer,
                        &hidden,
                        &prompt_embeds,
                        &img_ids,
                        &txt_ids,
                        ts,
                        Some(guidance),
                        target_seq,
                        attention_plan,
                        transformer_window,
                        &req.cancel,
                    );
                }
                // klein: distilled (CFG-free) or true-CFG via a negative pass when guidance > 1.
                let v = self.velocity(
                    transformer,
                    &hidden,
                    &prompt_embeds,
                    &img_ids,
                    &txt_ids,
                    ts,
                    None,
                    target_seq,
                    attention_plan,
                    transformer_window,
                    &req.cancel,
                )?;
                match &negative {
                    Some(neg) => {
                        let vn = self.velocity(
                            transformer,
                            &hidden,
                            neg,
                            &img_ids,
                            &txt_ids,
                            ts,
                            None,
                            target_seq,
                            attention_plan,
                            transformer_window,
                            &req.cancel,
                        )?;
                        // vn + guidance·(v − vn)
                        Ok((&vn + ((&v - &vn)? * guidance as f64)?)?)
                    }
                    None => Ok(v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
        // Decode the final latent: native FLUX.2 VAE by default, or the `flux2` PiD student (4× SR) when
        // this generation opted in (`req.use_pid`) and `with_pid` loaded one (sc-8044). Both take the same
        // unpacked latent and emit `[-1, 1]` pixels (PiD at 4×); `to_image` reads the size from the tensor.
        let pid_decoder = self.pid_decoder_for(req)?;
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&packed)?, // [1,3,4H,4W]
            None if self.memory.tile_vae_decode => vae.decode_packed_tiled(
                &packed,
                self.memory
                    .decode_tile_edge
                    .unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE),
                self.memory
                    .decode_overlap
                    .unwrap_or(crate::memory_strategy::DECODE_OVERLAP),
            )?,
            None => vae.decode_packed(&packed)?, // [1,3,H,W] in [-1,1]
        };
        to_image(&decoded)
    }

    /// Run the transformer on the joint `[target, refs]` image stream and keep the leading
    /// `target_seq` velocity tokens (the target image stream; `proj_out` is per-token, so the slice is
    /// exact). `guidance` is `Some(scale)` for dev (embedded guidance) and `None` for klein (distilled
    /// / true-CFG via the caller's negative pass).
    #[allow(clippy::too_many_arguments)]
    fn velocity(
        &self,
        transformer: &Flux2Transformer,
        hidden: &Tensor,
        embeds: &Tensor,
        img_ids: &[[i64; 4]],
        txt_ids: &[[i64; 4]],
        ts: f32,
        guidance: Option<f32>,
        target_seq: usize,
        attention_plan: AttentionPlan<'_>,
        transformer_window: usize,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        let out = transformer
            .forward_with_memory(
                hidden,
                embeds,
                img_ids,
                txt_ids,
                ts,
                guidance,
                attention_plan,
                transformer_window,
                cancel,
            )
            .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?;
        Ok(out.narrow(1, 0, target_seq)?)
    }

    /// Encode N reference images into packed transformer tokens + their grid ids. Each: Lanczos-resize
    /// to the render size → normalize to `[-1,1]` NCHW → [`Flux2Vae::encode_packed`] (the mean encode +
    /// 2×2 patchify + bn-normalize the transformer space expects) → pack to `[1, seq, 128]`, tagged
    /// with grid ids at `t = 10 + 10·i`. Returns the concatenated `([1, Σseq, 128], Σ grid ids)`.
    fn encode_references(
        &self,
        vae: &Flux2Vae,
        references: &[Image],
        width: u32,
        height: u32,
    ) -> Result<(Tensor, Vec<[i64; 4]>)> {
        let (lat_h, lat_w) = pipeline::latent_dims(width, height);
        let mut tokens: Vec<Tensor> = Vec::with_capacity(references.len());
        let mut ids: Vec<[i64; 4]> = Vec::with_capacity(references.len() * lat_h * lat_w);
        for (i, image) in references.iter().enumerate() {
            let nchw = preprocess_ref(image, width, height, &self.pipe.device, self.pipe.dtype)?;
            let packed = vae.encode_packed(&nchw)?; // [1, 128, H/16, W/16]
            tokens.push(pipeline::pack_nchw(&packed)?); // [1, seq, 128]
            ids.extend(pipeline::prepare_grid_ids_t(
                lat_h,
                lat_w,
                10 + 10 * i as i64,
            ));
        }
        Ok((Tensor::cat(&tokens, 1)?, ids))
    }
}

fn validate_base_binding(
    paths: &Flux2EditPaths,
    spec: &candle_gen::gen_core::LoadSpec,
) -> Result<()> {
    match &spec.weights {
        candle_gen::gen_core::WeightsSource::Dir(root) if root == &paths.root => Ok(()),
        candle_gen::gen_core::WeightsSource::Dir(root) => Err(CandleError::Msg(format!(
            "flux2 edit: runtime base {} differs from admitted base {}",
            paths.root.display(),
            root.display()
        ))),
        candle_gen::gen_core::WeightsSource::File(_) => Err(CandleError::Msg(
            "flux2 edit: admitted base must be the runtime snapshot directory".to_owned(),
        )),
    }
}

fn validate_memory_load_spec(
    variant: Flux2Variant,
    spec: &candle_gen::gen_core::LoadSpec,
) -> Result<()> {
    let mut unsupported = Vec::new();
    if !spec.adapters.is_empty() {
        unsupported.push("adapters");
    }
    if spec.control.is_some() {
        unsupported.push("control");
    }
    if !spec.extra_controls.is_empty() {
        unsupported.push("extra_controls");
    }
    if spec.ip_adapter.is_some() {
        unsupported.push("ip_adapter");
    }
    if spec.pid.is_some() {
        unsupported.push("pid");
    }
    if spec.identity.is_some() {
        unsupported.push("identity");
    }
    if !spec.components.is_empty() {
        unsupported.push("components");
    }
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(CandleError::Msg(format!(
            "{} edit memory route does not realize LoadSpec fields: {}",
            variant.id(),
            unsupported.join(", ")
        )))
    }
}

fn ensure_ordinary_generate_allowed(
    admitted_context: Option<&MemoryRunContext>,
    label: &str,
) -> Result<()> {
    if admitted_context.is_some() {
        Err(CandleError::Msg(format!(
            "{label}: admitted model requires generate_with_memory_context"
        )))
    } else {
        Ok(())
    }
}

fn validate_memory_authority(
    memory: GenerationMemory,
    admitted_context: Option<&MemoryRunContext>,
    label: &str,
) -> Result<()> {
    if memory != GenerationMemory::default() && admitted_context.is_none() {
        Err(CandleError::Msg(format!(
            "{label}: constrained memory requires an exact admitted context"
        )))
    } else {
        Ok(())
    }
}

fn validate_admitted_context(
    admitted: &MemoryRunContext,
    runtime: &MemoryRunContext,
    label: &str,
) -> Result<()> {
    if admitted != runtime {
        Err(CandleError::Msg(format!(
            "{label}: request memory context changed after provider load"
        )))
    } else {
        Ok(())
    }
}

fn validate_memory_request(
    context: &MemoryRunContext,
    req: &Flux2EditRequest,
    reference_count: usize,
) -> Result<()> {
    let expected_references = u32::try_from(reference_count).map_err(|_| {
        CandleError::Msg("flux2 edit: reference count exceeds the admitted domain".to_owned())
    })?;
    if context.geometry.width != req.width
        || context.geometry.height != req.height
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.reference_count != expected_references
        || !context.has_reference
        || context.use_pid != req.use_pid
    {
        return Err(CandleError::Msg(format!(
            "flux2 edit: request changed after admission (admitted={}x{} batch={} frames={} references={} has_reference={} use_pid={}; runtime={}x{} batch=1 frames=1 references={} has_reference=true use_pid={})",
            context.geometry.width,
            context.geometry.height,
            context.geometry.batch,
            context.geometry.frames,
            context.geometry.reference_count,
            context.has_reference,
            context.use_pid,
            req.width,
            req.height,
            expected_references,
            req.use_pid,
        )));
    }
    Ok(())
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard
/// (sc-8987, the sc-8646 bug class) mirrors the registered txt2img `validate` and the flux1 control
/// provider: `gen_core::TextTokenizer::tokenize("")` short-circuits to a (1, 0) encoding BEFORE the
/// chat template runs, so an empty prompt would reach the TE as a zero-length sequence and surface
/// as a deep tensor-shape error (or degenerate conditioning) instead of a clean validation error.
fn validate_request(req: &Flux2EditRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("flux2 edit: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "flux2 edit: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("flux2 edit: steps must be >= 1".into()));
    }
    Ok(())
}

/// Lanczos-resize a reference [`Image`] (RGB8) to the render size, normalize `[0,255] → [-1,1]`, lay
/// out as NCHW `[1, 3, H, W]` — the input [`Flux2Vae::encode_packed`] expects. Mirrors the mlx
/// `preprocess_ref_image` (`2·x − 1`). A no-op resize when the source is already the render size.
/// `pub(crate)` so the control provider ([`crate::control_provider`]) reuses it to VAE-encode the
/// pose/union control image (sc-7460).
pub(crate) fn preprocess_ref(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    if image.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "flux2 edit: reference pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    let resized: Vec<f32> = if (ih, iw) == (rh, rw) {
        image.pixels.iter().map(|&v| v as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, rh, rw)? // HWC f32 [0,255]
    };
    // [0,255] → [-1,1], then HWC → NCHW.
    let data: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let hwc = Tensor::from_vec(data, (rh, rw, 3), device)?;
    let nchw = hwc.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
    Ok(nchw.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admitted_edit_context(reference_count: u32) -> MemoryRunContext {
        let mut spec = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(PathBuf::from("/flux2-dev")),
        )
        .with_quant(Quant::Q4);
        spec.load_shape = candle_gen::gen_core::LoadShape::DeferredMaterialization;
        let contract = crate::memory_strategy::provider_contract(&spec).unwrap();
        candle_gen::gen_core::standard_memory_behavior_context(
            &contract,
            candle_gen::gen_core::MemoryStrategy::BoundedDecode,
            crate::memory_strategy::resolved_numeric_tier(&spec).unwrap(),
            candle_gen::gen_core::MemoryBehaviorRoute {
                mode: candle_gen::gen_core::MemoryMode::Edit,
                reference_count,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn admitted_edit_rejects_ordinary_generate_and_context_mutation() {
        let admitted = admitted_edit_context(2);
        assert!(ensure_ordinary_generate_allowed(Some(&admitted), "flux2 edit").is_err());
        assert!(ensure_ordinary_generate_allowed(None, "flux2 edit").is_ok());

        let mut mutated = admitted.clone();
        mutated.mode = candle_gen::gen_core::MemoryMode::Other("style_variations".to_owned());
        assert!(validate_admitted_context(&admitted, &mutated, "flux2 edit").is_err());
        assert!(validate_admitted_context(&admitted, &admitted, "flux2 edit").is_ok());
        let constrained = GenerationMemory {
            stage_residency: true,
            ..Default::default()
        };
        assert!(validate_memory_authority(constrained, None, "flux2 edit").is_err());
        assert!(validate_memory_authority(constrained, Some(&admitted), "flux2 edit").is_ok());
        assert!(validate_memory_authority(GenerationMemory::default(), None, "flux2 edit").is_ok());
    }

    #[test]
    fn admitted_edit_binds_geometry_reference_count_and_pid() {
        let context = admitted_edit_context(2);
        let mut request = Flux2EditRequest {
            prompt: "edit".to_owned(),
            ..Default::default()
        };
        assert!(validate_memory_request(&context, &request, 2).is_ok());
        request.width = 512;
        assert!(validate_memory_request(&context, &request, 2).is_err());
        request.width = 1024;
        assert!(validate_memory_request(&context, &request, 1).is_err());
        request.use_pid = true;
        assert!(validate_memory_request(&context, &request, 2).is_err());
    }

    #[test]
    fn admitted_edit_binds_the_runtime_base_path() {
        let paths = Flux2EditPaths {
            root: PathBuf::from("/runtime"),
        };
        let matching = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(paths.root.clone()),
        );
        assert!(validate_base_binding(&paths, &matching).is_ok());
        let mismatched = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(PathBuf::from("/admitted")),
        );
        assert!(validate_base_binding(&paths, &mismatched).is_err());
    }

    #[test]
    fn admitted_klein_edit_rejects_every_unrealized_load_spec_field() {
        use candle_gen::gen_core::{
            AdapterKind, AdapterSpec, IdentityWeights, PidWeights, WeightsSource,
        };

        let base =
            || candle_gen::gen_core::LoadSpec::new(WeightsSource::Dir(PathBuf::from("/klein")));
        assert!(validate_memory_load_spec(Flux2Variant::Klein9b, &base()).is_ok());

        let mut adapters = base();
        adapters.adapters.push(AdapterSpec::new(
            PathBuf::from("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        let mut control = base();
        control.control = Some(WeightsSource::File(PathBuf::from("control.safetensors")));
        let mut extra_controls = base();
        extra_controls
            .extra_controls
            .push(WeightsSource::File(PathBuf::from(
                "extra-control.safetensors",
            )));
        let mut ip_adapter = base();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir(PathBuf::from("ip-adapter")));
        let mut pid = base();
        pid.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(PathBuf::from("pid.safetensors")),
            gemma: WeightsSource::Dir(PathBuf::from("gemma")),
        });
        let mut identity = base();
        identity.identity = Some(IdentityWeights::default());
        let mut text_encoder = base();
        text_encoder.text_encoder = Some(WeightsSource::Dir(PathBuf::from("external-te")));
        assert!(
            validate_memory_load_spec(Flux2Variant::Klein9b, &text_encoder).is_ok(),
            "typed encoder substitutions are realized by the edit memory route"
        );
        let components = base().with_component(
            "unwired_component",
            WeightsSource::File(PathBuf::from("component.safetensors")),
        );

        for (field, spec) in [
            ("adapters", adapters),
            ("control", control),
            ("extra_controls", extra_controls),
            ("ip_adapter", ip_adapter),
            ("pid", pid),
            ("identity", identity),
            ("components", components),
        ] {
            let error = validate_memory_load_spec(Flux2Variant::Klein9b, &spec)
                .expect_err("unrealized LoadSpec field must fail before weight load");
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn legacy_edit_constructor_rejects_constrained_memory_without_context() {
        let paths = Flux2EditPaths {
            root: PathBuf::from("/missing-flux2-dev"),
        };
        let error = Flux2Edit::load_dev_with_memory(
            &paths,
            Some(Quant::Q4),
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
        )
        .err()
        .expect("legacy constrained edit load must fail");
        assert!(error.to_string().contains("exact admitted context"));
    }

    #[test]
    fn edit_spec_loader_consumes_authored_text_encoder_before_device_load() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("base");
        let selected = tmp.path().join("selected");
        std::fs::create_dir_all(&root).unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &selected,
            Flux2Variant::Dev.encoder_contract(),
        )
        .unwrap();
        let config_path = selected.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["hidden_size"] = serde_json::json!(7);
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let paths = Flux2EditPaths { root: root.clone() };
        let spec =
            candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(root))
                .with_text_encoder(candle_gen::gen_core::WeightsSource::Dir(selected));
        let error =
            Flux2Edit::load_dev_with_memory_spec(&paths, None, &spec, GenerationMemory::default())
                .err()
                .expect("authored wrong-shape encoder must reject before device load")
                .to_string();
        assert!(error.contains("field hidden_size"), "{error}");
    }

    #[test]
    fn planning_without_tokenizer_is_metadata_only_but_load_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("base");
        let selected = tmp.path().join("selected-component");
        std::fs::create_dir_all(&root).unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &selected,
            Flux2Variant::Dev.encoder_contract(),
        )
        .unwrap();
        let paths = Flux2EditPaths { root: root.clone() };
        let spec =
            candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(root))
                .with_text_encoder(candle_gen::gen_core::WeightsSource::Dir(selected));

        crate::memory_strategy::provider_contract(&spec)
            .expect("memory planning must not require a runtime tokenizer receipt");
        let error =
            Flux2Edit::load_dev_with_memory_spec(&paths, None, &spec, GenerationMemory::default())
                .err()
                .expect("actual load must require the retained tokenizer receipt")
                .to_string();
        assert!(error.contains("no retained tokenizer artifact"), "{error}");
    }

    /// The request defaults match the klein edit production knobs (1024², 4 distilled steps, CFG-free).
    #[test]
    fn request_defaults() {
        let r = Flux2EditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS as usize);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard (sc-8987, sc-8646 bug class): an empty or whitespace-only prompt is a
    /// clean validation error, never a zero-length TE sequence; a real prompt passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Flux2EditRequest::default();
        let err = validate_request(&empty).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let whitespace = Flux2EditRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(&whitespace).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let ok = Flux2EditRequest {
            prompt: "a portrait".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards moved into `validate_request` still fire (no regression from the
    /// sc-8987 refactor).
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Flux2EditRequest {
            prompt: "a portrait".into(),
            width: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Flux2EditRequest {
            prompt: "a portrait".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    /// `preprocess_ref` lays a same-size RGB8 reference out as NCHW `[1,3,H,W]` in `[-1,1]`: white → 1,
    /// black → −1 (the `2·x − 1` normalization), with the channel axis moved to front.
    #[test]
    fn preprocess_ref_normalizes_and_lays_out_nchw() {
        let dev = Device::Cpu;
        // 2×2 image: top-left white, the rest black.
        let pixels = vec![
            255, 255, 255, 0, 0, 0, // row 0: white, black
            0, 0, 0, 0, 0, 0, // row 1: black, black
        ];
        let img = Image {
            width: 2,
            height: 2,
            pixels,
        };
        let t = preprocess_ref(&img, 2, 2, &dev, DType::F32).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        // Channel 0 (R), row-major after the HWC→NCHW move: [1, −1, −1, −1].
        let r = t
            .narrow(1, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(r, vec![1.0, -1.0, -1.0, -1.0]);
    }
}
