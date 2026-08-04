//! FLUX.2-dev **strict-pose ControlNet** provider (sc-7460, epic 6564) — the candle (Windows/CUDA)
//! sibling of mlx-gen-flux2's `flux2_dev_control` (sc-2292). Strict pose (and the union's
//! canny/depth/hed/mlsd/scribble/gray) on FLUX.2-dev via `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`,
//! a VACE-style control branch overlaid on the dev base DiT.
//!
//! **How it conditions:** the pose/union control image is VAE-encoded into the packed transformer
//! latent ([`Flux2Vae::encode_packed`]), packed to `[1, seq, 128]`, then concatenated with a zero
//! inpaint **mask** (4) + a zero **inpaint latent** (128) → the 260-ch control context (the union
//! ControlNet's pose-only channel layout). [`Flux2ControlTransformer`] runs the parity-proven dev DiT
//! plus the control branch ([`Flux2ControlBranch`]): per-block hints are computed once from the
//! post-embedder streams and added into the base image stream after base double blocks `[0, 2, 4, 6]`,
//! scaled by `control_scale` (dev README sweet spot 0.65–0.80). The control context is clean + constant
//! across the denoise (encoded once). dev is guidance-distilled — a single embedded-guidance forward,
//! no true-CFG / negative pass.
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::edit_provider`] / the SDXL control providers). The dev base
//! is the 32B flagship, so it loads via the CPU-stage → quantize-onto-GPU path ([`crate::quant`]); the
//! ~8 GB bf16 control overlay loads dense on the device and quantizes in place. Determinism is the
//! candle-lane contract (sc-3673): the seeded CPU init noise reuses [`pipeline::create_noise`].

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    GenerationMemory, Image, MemoryProviderContract, MemoryRunContext, PidWeights, PreviewSink,
    Progress, Quant,
};
// `LatentDecoder` brings the `PidDecoder::decode` trait method into scope (sc-8044).
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::{PidDecoder, PidEngine};

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE_DEV, DEFAULT_STEPS_DEV, SIZE_MULTIPLE};
use crate::edit_provider::preprocess_ref;
use crate::text_encoder::Flux2PromptEncoder;
use crate::transformer::{Flux2ControlBranch, Flux2ControlTransformer};
use crate::vae::Flux2Vae;
use crate::{pipeline, to_image, Pipeline, PID_BACKBONE};

/// Default `control_context_scale` — the dev Fun-Controlnet-Union README sweet spot is 0.65–0.80; the
/// mlx worker defaults to 0.75. Strong pose lock without over-constraining the base.
pub const DEFAULT_CONTROL_SCALE: f32 = 0.75;

/// Paths to the FLUX.2-dev control checkpoints: the dev snapshot dir (`text_encoder/`, `transformer/`,
/// `vae/`, `tokenizer/`) + the Fun-Controlnet-Union overlay (a single `.safetensors` file, e.g.
/// `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors`, or a dir holding it).
pub struct Flux2ControlPaths {
    /// FLUX.2-dev diffusers snapshot dir.
    pub root: PathBuf,
    /// The Fun-Controlnet-Union control checkpoint (`.safetensors` file or a dir containing it).
    pub control: PathBuf,
}

/// One FLUX.2-dev strict-pose control request. dev is guidance-distilled — `guidance` is the embedded
/// scalar (single forward, no negative prompt).
#[derive(Clone)]
pub struct Flux2ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Embedded guidance scale (dev default ≈ 4.0).
    pub guidance: f32,
    /// `control_context_scale` — how strongly the control branch locks the base (≈ 0.65–0.80).
    pub control_scale: f32,
    pub seed: u64,
    /// Opt into the PiD super-resolving decoder (epic 7840, sc-8044): when `true` **and** the model was
    /// loaded with [`with_pid`](Flux2Control::with_pid), the final latent is decoded by the `flux2` PiD
    /// student (4× SR → 2K/4K) instead of the native FLUX.2 VAE. `false` (default) keeps the VAE decode.
    pub use_pid: bool,
    /// Per-step latent-preview sink (epic 16948, sc-16955) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// which this provider cannot carry because it is worker-invoked by name rather than through a
    /// registered descriptor. Default is inert, and an inert sink makes a seeded render
    /// byte-identical to one with no preview at all.
    pub preview: PreviewSink,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Flux2ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: DEFAULT_STEPS_DEV as usize,
            guidance: DEFAULT_GUIDANCE_DEV,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 0,
            use_pid: false,
            preview: PreviewSink::default(),
            cancel: CancelFlag::default(),
        }
    }
}

/// A loaded FLUX.2-dev control model: the Mistral text encoder + the dev DiT wrapped in its control
/// branch ([`Flux2ControlTransformer`]) + the VAE **with the encoder** (the control-image encode).
/// `generate` takes `&self` (no per-call mutation), so one load serves many renders.
pub struct Flux2Control {
    pipe: Pipeline,
    te: Option<Flux2PromptEncoder>,
    /// Prompt tokenizer, loaded+parsed **once** at load and reused across encodes (sc-8991 / F-011)
    /// instead of re-parsing `tokenizer.json` per prompt.
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    transformer: Option<Flux2ControlTransformer>,
    vae: Option<Flux2Vae>,
    control_path: PathBuf,
    quant: Option<Quant>,
    memory: GenerationMemory,
    memory_contract: Option<MemoryProviderContract>,
    admitted_context: Option<MemoryRunContext>,
    lifecycle: std::sync::Mutex<()>,
    /// Optional PiD super-resolving decoder (epic 7840, sc-8044), attached via [`with_pid`](Self::with_pid).
    /// FLUX.2 control composes the FLUX.2 VAE, so it loads the SAME `flux2` student ([`PID_BACKBONE`]) as
    /// the registered FLUX.2 provider.
    pid: Option<PidEngine>,
}

impl Flux2Control {
    /// Load the dev base (CPU-stage → quantize-onto-GPU when `quant` is set; dense otherwise — fixture
    /// only, the 32B does not fit dense), the Fun-Controlnet-Union control overlay (dense on the device
    /// → quantized in place), and the VAE with its encoder. `quant` (Q4/Q8) is required in practice for
    /// the real 32B weights.
    pub fn load(paths: &Flux2ControlPaths, quant: Option<Quant>) -> Result<Self> {
        Self::load_with_memory(paths, quant, GenerationMemory::default())
    }

    pub fn load_with_memory(
        paths: &Flux2ControlPaths,
        quant: Option<Quant>,
        memory: GenerationMemory,
    ) -> Result<Self> {
        let mut spec = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(paths.root.clone()),
        );
        spec.quantize = quant;
        let loaded_quant = crate::memory_strategy::resolved_quant(&spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        Self::load_bound(paths, loaded_quant, memory, None, None)
    }

    fn load_bound(
        paths: &Flux2ControlPaths,
        loaded_quant: Option<Quant>,
        memory: GenerationMemory,
        memory_contract: Option<MemoryProviderContract>,
        admitted_context: Option<MemoryRunContext>,
    ) -> Result<Self> {
        validate_memory_authority(memory, admitted_context.as_ref(), "flux2 control")?;
        let optimized =
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks;
        if optimized && !memory.stage_residency {
            return Err(CandleError::Msg(
                "flux2 control: optimized memory rungs require staged residency".into(),
            ));
        }
        let device = candle_gen::default_device()?;
        // PiD (super-resolving decode) is wired only through the txt2img render path (epic 7840 /
        // sc-7853); the control provider passes `None`.
        let pipe = Pipeline::load(Flux2Variant::Dev, loaded_quant, &paths.root, &device, None);

        // Base DiT + Mistral TE. Packed MLX tier → build directly on the GPU from the packed parts
        // (sc-9087, no ~105 GB dense CPU staging); dense tier → stage dense in CPU RAM and quantize each
        // projection onto the GPU. Shared TE+DiT loader with txt2img / edit (F-024, sc-9004); the control
        // branch overlay below (`Flux2ControlBranch` → `Flux2ControlTransformer`) is the per-site addition.
        let (te, transformer, vae) = if memory.stage_residency {
            (None, None, None)
        } else {
            let (te, base) = pipe.load_te_and_dit()?;

            // The control overlay is small (~8 GB bf16) and fits on the GPU; load it dense on-device and
            // quantize in place (the 260-ch `control_img_in` stays dense — 260 ∤ 32).
            let control_vb = control_var_builder(&paths.control, pipe.dtype, &device)?;
            let mut branch = Flux2ControlBranch::new(&pipe.cfg, control_vb)?;
            if let Some(q) = loaded_quant {
                branch.quantize(q, &device)?;
            }
            let transformer = Flux2ControlTransformer::new(base, branch);

            let vae = Flux2Vae::new_with_encoder(pipe.component_vb("vae")?)?;
            (Some(te), Some(transformer), Some(vae))
        };
        let tokenizer = pipe.build_tokenizer()?;
        Ok(Self {
            pipe,
            te,
            tokenizer,
            transformer,
            vae,
            control_path: paths.control.clone(),
            quant: loaded_quant,
            memory,
            memory_contract,
            admitted_context,
            lifecycle: std::sync::Mutex::new(()),
            pid: None,
        })
    }

    pub fn load_with_memory_context(
        paths: &Flux2ControlPaths,
        quant: Option<Quant>,
        spec: &candle_gen::gen_core::LoadSpec,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> Result<Self> {
        validate_admitted_paths(paths, spec)?;
        let loaded_quant = crate::memory_strategy::resolved_quant(spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        let contract = crate::memory_strategy::provider_contract(spec)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        crate::memory_strategy::validate_context(&contract, context, loaded_quant)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        if quant.is_some() && quant != loaded_quant {
            return Err(CandleError::Msg(format!(
                "flux2 control: requested {quant:?} but the admitted snapshot resolves to {loaded_quant:?}"
            )));
        }
        let memory = contract
            .generation_memory(&context.selection)
            .unwrap_or_default();
        Self::load_bound(
            paths,
            loaded_quant,
            memory,
            Some(contract),
            Some(context.clone()),
        )
    }

    /// Attach the optional PiD super-resolving decoder (epic 7840, sc-8044). Same [`PidWeights`] load-spec
    /// as the registry FLUX.2 provider; control composes the FLUX.2 VAE so it loads the **same**
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
    fn pid_decoder_for(&self, req: &Flux2ControlRequest) -> Result<Option<PidDecoder>> {
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
            "flux2 control",
            0.0,
        )
    }

    /// Generate one strict-pose-conditioned image. `control_image` is the pose/union skeleton (the
    /// worker pre-fits it to the render size; this re-resizes defensively).
    pub fn generate(
        &self,
        req: &Flux2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        crate::run_bespoke_request(
            &self.lifecycle,
            || {
                ensure_ordinary_generate_allowed(self.admitted_context.as_ref(), "flux2 control")?;
                validate_memory_authority(
                    self.memory,
                    self.admitted_context.as_ref(),
                    "flux2 control",
                )?;
                self.generate_inner(req, control_image, on_progress)
            },
            || self.pipe.device.synchronize(),
        )
    }

    /// Execute the exact control request admitted at load. Context identity, geometry, PiD, and the
    /// certified overlay binding are rechecked immediately before the serialized request.
    pub fn generate_with_memory_context(
        &self,
        context: &MemoryRunContext,
        req: &Flux2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        crate::run_bespoke_request(
            &self.lifecycle,
            || {
                let admitted = self.admitted_context.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "flux2 control: model was not loaded with a memory context".to_owned(),
                    )
                })?;
                let contract = self.memory_contract.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "flux2 control: admitted model lost its memory contract".to_owned(),
                    )
                })?;
                validate_admitted_context(admitted, context, "flux2 control")?;
                validate_memory_request(context, req)?;
                crate::memory_strategy::validate_context(contract, context, self.quant)
                    .map_err(|error| CandleError::Msg(error.to_string()))?;
                self.generate_inner(req, control_image, on_progress)
            },
            || self.pipe.device.synchronize(),
        )
    }

    fn generate_inner(
        &self,
        req: &Flux2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        validate_request(req)?;

        if self.memory.stage_residency && req.use_pid {
            return Err(CandleError::Msg(
                "flux2 control: optimized native-VAE memory rungs do not support PiD".into(),
            ));
        }

        let device = &self.pipe.device;
        let cfg = &self.pipe.cfg;

        // Prompt embeds (text-only Mistral) + the packed 260-ch control context are seed-independent:
        // encode once.
        let prompt_embeds = if let Some(te) = self.te.as_ref() {
            self.pipe.encode(te, &self.tokenizer, &req.prompt)?
        } else {
            candle_gen::check_cancel(&req.cancel)?;
            let te = self.pipe.load_te_seq()?;
            let result = self.pipe.encode(&te, &self.tokenizer, &req.prompt);
            let result = candle_gen::synchronize_result(&self.pipe.device, result);
            drop(te);
            result?
        };
        let control_context = if let Some(vae) = self.vae.as_ref() {
            self.encode_control_context(vae, control_image, req.width, req.height)?
        } else {
            candle_gen::check_cancel(&req.cancel)?;
            let vae = Flux2Vae::new_with_encoder(self.pipe.component_vb("vae")?)?;
            let result = self.encode_control_context(&vae, control_image, req.width, req.height);
            let result = candle_gen::synchronize_result(&self.pipe.device, result);
            drop(vae);
            result?
        };

        let staged_transformer = if self.transformer.is_none() {
            candle_gen::check_cancel(&req.cancel)?;
            let base = self
                .pipe
                .load_dit_seq_with_memory(self.memory.stream_transformer_blocks)?;
            let control_vb =
                control_var_builder(&self.control_path, self.pipe.dtype, &self.pipe.device)?;
            let mut branch = Flux2ControlBranch::new(&self.pipe.cfg, control_vb)?;
            if let Some(quant) = self.quant {
                branch.quantize(quant, &self.pipe.device)?;
            }
            Some(Flux2ControlTransformer::new(base, branch))
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
        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(cfg.max_sequence_length);

        // Curated sampler/scheduler routing (epic 7114 P4) — the same driver the txt2img/edit paths use.
        // No per-request sampler/scheduler knob, so this runs the default (`None`) euler over the native
        // empirical-mu schedule (the N1 no-op reproducing the legacy flow-match loop within tolerance).
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), req.steps);
        let native = pipeline::schedule(req.steps, req.width, req.height);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);

        let latents = pipeline::create_noise(cfg, req.seed, req.width, req.height, device)?;
        // Per-step latent preview (epic 16948, sc-16955), bound to the same `(lat_h, lat_w)` the decode
        // tail below unpacks against. `control_context` is a closure capture, constant across steps and
        // never part of the running latent, so the control hint cannot reach a frame.
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
        // The driver does cancel + progress + the integrator step. The control forward lives inside the
        // predict closure so a multi-eval solver re-runs it. dev is guidance-distilled: a single forward
        // feeding the embedded guidance scalar (no negative pass). FLUX.2 embeds σ×1000.
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
                Ok(transformer
                    .forward_with_memory(
                        latents,
                        &prompt_embeds,
                        &img_ids,
                        &txt_ids,
                        ts,
                        Some(req.guidance),
                        &control_context,
                        req.control_scale,
                        attention_plan,
                        transformer_window,
                        &req.cancel,
                    )
                    .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))?)
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

    /// Build the packed control context `[1, seq, 260]` from the pose/union control image (the fork's
    /// `pipeline_flux2_control.py`): VAE-encode → 2×2 patchify → BN-normalize → pack (`control_latents`,
    /// 128), concatenated with a zero inpaint **mask** (4) + a zero **inpaint latent** (128). For pure
    /// pose (no inpaint image / mask) the fork's mask + inpaint latent are all-zero. `seq` equals the
    /// target latent sequence (built at the same `width`/`height`), so the control context aligns 1:1
    /// with the base image tokens.
    fn encode_control_context(
        &self,
        vae: &Flux2Vae,
        image: &Image,
        width: u32,
        height: u32,
    ) -> Result<Tensor> {
        let cfg = &self.pipe.cfg;
        let device = &self.pipe.device;
        let nchw = preprocess_ref(image, width, height, device, self.pipe.dtype)?;
        let packed = vae.encode_packed(&nchw)?; // [1, 128, H/16, W/16]
        let control_packed = pipeline::pack_nchw(&packed)?; // [1, seq, 128]
        let (_, seq, _) = control_packed.dims3()?;
        // Union pose-only layout: zero mask (in_channels / num_latent_channels = 128/32 = 4, the 2×2
        // patch) + zero inpaint latent (in_channels, 128) → concatenated on the channel axis = 260.
        let in_ch = cfg.in_channels;
        let mask_ch = in_ch / cfg.num_latent_channels;
        let mask = Tensor::zeros((1, seq, mask_ch), DType::F32, device)?;
        let inpaint = Tensor::zeros((1, seq, in_ch), DType::F32, device)?;
        let cc = Tensor::cat(&[&control_packed, &mask, &inpaint], 2)?;
        debug_assert_eq!(
            cc.dim(2)?,
            crate::transformer::CONTROL_IN_DIM,
            "control context must be 260ch"
        );
        Ok(cc)
    }
}

fn source_path(source: &candle_gen::gen_core::WeightsSource) -> &Path {
    match source {
        candle_gen::gen_core::WeightsSource::Dir(path)
        | candle_gen::gen_core::WeightsSource::File(path) => path,
    }
}

fn validate_admitted_paths(
    paths: &Flux2ControlPaths,
    spec: &candle_gen::gen_core::LoadSpec,
) -> Result<()> {
    match &spec.weights {
        candle_gen::gen_core::WeightsSource::Dir(root) if root == &paths.root => {}
        candle_gen::gen_core::WeightsSource::Dir(root) => {
            return Err(CandleError::Msg(format!(
                "flux2 control: runtime base {} differs from admitted base {}",
                paths.root.display(),
                root.display()
            )));
        }
        candle_gen::gen_core::WeightsSource::File(_) => {
            return Err(CandleError::Msg(
                "flux2 control: admitted base must be the runtime snapshot directory".to_owned(),
            ));
        }
    }
    let admitted_control = spec.control.as_ref().ok_or_else(|| {
        CandleError::Msg("flux2 control: admission spec has no control overlay".to_owned())
    })?;
    if source_path(admitted_control) != paths.control {
        return Err(CandleError::Msg(format!(
            "flux2 control: runtime overlay {} differs from admitted overlay {}",
            paths.control.display(),
            source_path(admitted_control).display()
        )));
    }
    if !paths.control.exists() {
        return Err(CandleError::Msg(format!(
            "flux2 control: admitted overlay {} is missing",
            paths.control.display()
        )));
    }
    let overlay_bytes = candle_gen::gen_core::weightsmeta::safetensors_path_bytes(&paths.control);
    if overlay_bytes == 0 {
        return Err(CandleError::Msg(format!(
            "flux2 control: admitted overlay {} has zero safetensors bytes",
            paths.control.display()
        )));
    }
    Ok(())
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

fn validate_memory_request(context: &MemoryRunContext, req: &Flux2ControlRequest) -> Result<()> {
    if context.geometry.width != req.width
        || context.geometry.height != req.height
        || context.geometry.batch != 1
        || context.geometry.frames != 1
        || context.geometry.reference_count != 0
        || context.has_reference
        || context.use_pid != req.use_pid
    {
        return Err(CandleError::Msg(format!(
            "flux2 control: request changed after admission (admitted={}x{} batch={} frames={} references={} has_reference={} use_pid={}; runtime={}x{} batch=1 frames=1 references=0 has_reference=false use_pid={})",
            context.geometry.width,
            context.geometry.height,
            context.geometry.batch,
            context.geometry.frames,
            context.geometry.reference_count,
            context.has_reference,
            context.use_pid,
            req.width,
            req.height,
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
fn validate_request(req: &Flux2ControlRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("flux2 control: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "flux2 control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("flux2 control: steps must be >= 1".into()));
    }
    Ok(())
}

/// Open a VarBuilder over the Fun-Controlnet-Union checkpoint — a single `.safetensors` `File` or a
/// `Dir` containing the `.safetensors` shards — on `device` at `dtype`.
fn control_var_builder(path: &Path, dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    candle_gen::load_path_mmap(path, dtype, device, "flux2 control")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admitted_control_context() -> MemoryRunContext {
        let mut spec = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(PathBuf::from("/flux2-dev")),
        )
        .with_quant(Quant::Q4)
        .with_control(candle_gen::gen_core::WeightsSource::File(PathBuf::from(
            "/control.safetensors",
        )));
        spec.load_shape = candle_gen::gen_core::LoadShape::DeferredMaterialization;
        let contract = crate::memory_strategy::provider_contract(&spec).unwrap();
        candle_gen::gen_core::standard_memory_behavior_context(
            &contract,
            candle_gen::gen_core::MemoryStrategy::BoundedDecode,
            crate::memory_strategy::resolved_numeric_tier(&spec).unwrap(),
            candle_gen::gen_core::MemoryBehaviorRoute {
                mode: candle_gen::gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: Some(crate::memory_strategy::CONTROL_OVERLAY.to_owned()),
            },
        )
        .unwrap()
    }

    #[test]
    fn admitted_control_rejects_ordinary_generate_and_context_mutation() {
        let admitted = admitted_control_context();
        assert!(ensure_ordinary_generate_allowed(Some(&admitted), "flux2 control").is_err());
        assert!(ensure_ordinary_generate_allowed(None, "flux2 control").is_ok());
        let mut mutated = admitted.clone();
        mutated.overlay = None;
        assert!(validate_admitted_context(&admitted, &mutated, "flux2 control").is_err());
        assert!(validate_admitted_context(&admitted, &admitted, "flux2 control").is_ok());
        let constrained = GenerationMemory {
            tile_vae_decode: true,
            ..Default::default()
        };
        assert!(validate_memory_authority(constrained, None, "flux2 control").is_err());
        assert!(validate_memory_authority(constrained, Some(&admitted), "flux2 control").is_ok());
        assert!(
            validate_memory_authority(GenerationMemory::default(), None, "flux2 control").is_ok()
        );
    }

    #[test]
    fn admitted_control_binds_geometry_and_pid() {
        let context = admitted_control_context();
        let mut request = Flux2ControlRequest {
            prompt: "pose".to_owned(),
            ..Default::default()
        };
        assert!(validate_memory_request(&context, &request).is_ok());
        request.height = 512;
        assert!(validate_memory_request(&context, &request).is_err());
        request.height = 1024;
        request.use_pid = true;
        assert!(validate_memory_request(&context, &request).is_err());
    }

    #[test]
    fn admitted_control_exact_binds_nonempty_overlay() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sc15833_flux2_control_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&root).unwrap();
        let overlay = root.join("control.safetensors");
        std::fs::write(&overlay, b"nonempty").unwrap();
        let paths = Flux2ControlPaths {
            root: root.clone(),
            control: overlay.clone(),
        };
        let matching = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(root.clone()),
        )
        .with_control(candle_gen::gen_core::WeightsSource::File(overlay.clone()));
        assert!(validate_admitted_paths(&paths, &matching).is_ok());

        let absent = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            root.clone(),
        ));
        assert!(validate_admitted_paths(&paths, &absent).is_err());

        let mismatch = matching
            .clone()
            .with_control(candle_gen::gen_core::WeightsSource::File(
                root.join("other.safetensors"),
            ));
        assert!(validate_admitted_paths(&paths, &mismatch).is_err());

        std::fs::remove_file(&overlay).unwrap();
        assert!(validate_admitted_paths(&paths, &matching).is_err());
        std::fs::write(&overlay, []).unwrap();
        assert!(validate_admitted_paths(&paths, &matching).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_control_constructor_rejects_constrained_memory_without_context() {
        let paths = Flux2ControlPaths {
            root: PathBuf::from("/missing-flux2-dev"),
            control: PathBuf::from("/missing-control.safetensors"),
        };
        let error = Flux2Control::load_with_memory(
            &paths,
            Some(Quant::Q4),
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            },
        )
        .err()
        .expect("legacy constrained control load must fail");
        assert!(error.to_string().contains("exact admitted context"));
    }

    /// The request defaults match the dev control production knobs (1024², 28 steps, guidance 4.0,
    /// control scale 0.75).
    #[test]
    fn request_defaults() {
        let r = Flux2ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, DEFAULT_STEPS_DEV as usize);
        assert_eq!(r.guidance, DEFAULT_GUIDANCE_DEV);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard (sc-8987, sc-8646 bug class): an empty or whitespace-only prompt is a
    /// clean validation error, never a zero-length TE sequence; a real prompt passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Flux2ControlRequest::default();
        let err = validate_request(&empty).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let whitespace = Flux2ControlRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        let err = validate_request(&whitespace).unwrap_err();
        assert!(err.to_string().contains("prompt is required"), "{err}");

        let ok = Flux2ControlRequest {
            prompt: "a dancer mid-leap".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards moved into `validate_request` still fire (no regression from the
    /// sc-8987 refactor).
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Flux2ControlRequest {
            prompt: "a dancer".into(),
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Flux2ControlRequest {
            prompt: "a dancer".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }
}
