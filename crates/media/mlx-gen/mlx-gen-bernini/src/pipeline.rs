//! The native Bernini renderer provider (sc-4706): loads the converted dual-expert snapshot
//! (sc-4705) + the stock Wan2.2 UMT5/VAE/tokenizer, and runs the boundary-switched, APG-guided
//! denoise in **spatial latent space**, decoding to an image (1 frame) or video.
//!
//! Mirrors `mlx_gen_wan::model::Wan14b`'s staging (UMT5 → experts → VAE) to bound peak memory, with
//! the dual-expert CFG `denoise_moe` replaced by `denoise_bernini` (per-step
//! [`crate::forward::guided_velocity`] over the resolved [`Mode`]).

use std::path::PathBuf;

use mlx_rs::memory::clear_cache;
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use mlx_gen::block_residency::BlockPlan;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result,
    SizeFloor, WeightsSource,
};
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{align_dim, decode_to_frames, frames_to_images, latent_shape};
use mlx_gen_wan::scheduler::{make_scheduler, SolverKind};
use mlx_gen_wan::text_encoder::{load_tokenizer, Umt5Encoder};
use mlx_gen_wan::{WanBlockStream, WanTransformer};

use crate::config::{resolve_mode, validate_bernini_geometry, BerniniKnobs, Defaults};
use crate::forward::{
    guided_velocity, num_momentum_buffers, vit_one_step, GuidanceParams, Mode, PackedForward,
    Trunk, VitGuidanceParams, VitMode, VitStreams,
};
use crate::guidance::MomentumBuffer;
use crate::preprocess::{encode_image, encode_videoclip};
use crate::{decoded_output_geometry, ProviderVae, PROVIDER_VAE_STRIDE};

pub const MODEL_ID: &str = "bernini_renderer";

/// Stable identity + advertised capabilities for the Bernini renderer (Wan2.2-A14B dual-expert with
/// source-id rotary + token-packed conditioning + APG guidance; t2v/t2i/i2i/v2v/r2v/rv2v).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "bernini",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Source media: a single Reference (i2i) / MultiReference (r2v refs) / VideoClip (v2v/rv2v
            // source video). Text-only modes (t2i/t2v) need no conditioning.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
            ],
            // LoRA/quant are follow-ons (sc-5146); the renderer ships dense bf16.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // The renderer is structurally always-staged (epic 10834, sc-10840): `generate_impl` holds
            // NO component weights on the generator and loads per generate in phase order — UMT5-XXL T5
            // → drop → (source-VAE encoder for i2i/v2v/r2v) → drop → the two co-resident MoE experts +
            // z16 VAE — dropping BOTH encoders (+ `clear_cache()`) before the experts, so peak unified
            // memory is already bounded to the dominant expert phase. The shared per-component footprint
            // reports the two experts as the DiT phase (the peak), so the fit-gate's staged estimate is
            // sound. `OffloadPolicy` is not consumed — there is no Resident-warm mode to toggle, so this
            // is unconditional physical staging rather than a selectable offload control.
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: true,
            supports_preview: false,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
}

/// The loaded Bernini renderer: resolved Wan2.2 config + Bernini knobs + the snapshot dir. The heavy
/// components are staged inside `generate_impl`.
pub struct BerniniRenderer {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    knobs: BerniniKnobs,
    root: PathBuf,
    quant: Option<Quant>,
}

/// Load the Bernini renderer from a converted MLX snapshot directory
/// ([`mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`] output: `low/high_noise_model` +
/// `t5_encoder` + `vae` + `tokenizer.json` + `config.json` + `bernini_renderer.json`).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "bernini_renderer: expected a model directory (converted MLX snapshot), not a single file".into(),
            ))
        }
    };
    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "bernini_renderer: config.json is not a dual-expert model (model_type={}); expected the \
             assembled Bernini renderer snapshot",
            config.model_type
        )));
    }
    let knobs = BerniniKnobs::from_dir(&root)?;
    Ok(Box::new(BerniniRenderer {
        descriptor: descriptor(),
        config,
        knobs,
        root,
        quant: spec.quantize,
    }))
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split. Bernini is
/// a FLAT snapshot — the renderer components are individual root-level FILES, not diffusers subdirs: the
/// UMT5 text encoder `t5_encoder.safetensors`, the WAN dual experts `low_noise_model.safetensors` +
/// `high_noise_model.safetensors`, and the z16 VAE `vae.safetensors`. A name-guessing consumer would
/// read the encoder as ZERO (no `text_encoder/` subdir at all); this seam reports the real file bytes.
/// The full `bernini` id additionally loads the Qwen2.5-VL planner files; those are not part of the
/// TE/DiT/VAE render split (they are still in the worker's whole-model total). Shared by `bernini` +
/// `bernini_renderer`.
///
/// Both ids declare unconditional staged residency — each is structurally always-staged
/// (the encoders are dropped + `clear_cache()`d before the two co-resident experts load), and this split
/// is the staged peak the fit-gate bounds (`max(encoders, DiT+VAE)`, dominated by the experts). The
/// worker can consume that physical fact independently of the false selectable-control bit and this
/// split generically through the registered footprint seam (this crate reports the bytes; the worker
/// decides how to use them).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["t5_encoder.safetensors"],
        &[
            "low_noise_model.safetensors",
            "high_noise_model.safetensors",
        ],
        &["vae.safetensors"],
    )
}

mlx_gen::register_generators! {
    pub(crate) const RENDERER_REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// One expert (high or low) with its prepared per-expert cross-attention K/V for the cond / empty-neg
/// text contexts (text embedding is per-expert, so K/V is built per expert).
struct BExpert<'a> {
    trunk: Trunk<'a>,
    cross_kv_cond: Vec<(Array, Array)>,
    cross_kv_uncond: Vec<(Array, Array)>,
}

impl<'a> BExpert<'a> {
    /// Build this expert's cross-attention K/V.
    ///
    /// On a rung-4 (windowed) trunk the DiT holds no blocks, so `prepare_cross_kv` cannot serve the
    /// cache from it; the stream walks the same plan once here instead. The caches are small
    /// (`[B, n, text_len, d]` per block) and are reused across every step and every guidance pass, so
    /// they stay resident deliberately — re-deriving them per window per forward would multiply a
    /// once-per-generate cost by the pass count for no residency saving.
    fn build(trunk: Trunk<'a>, context: &Array, context_null: &Array) -> Result<Self> {
        let dit = trunk.dit;
        let cc = dit.embed_text(context)?;
        let cu = dit.embed_text(context_null)?;
        let prepare = |embedded: &Array| -> Result<Vec<(Array, Array)>> {
            match trunk.window {
                None => dit.prepare_cross_kv(embedded),
                Some((stream, plan, cancel)) => {
                    stream.prepare_cross_kv_windowed(embedded, plan, cancel)
                }
            }
        };
        Ok(Self {
            cross_kv_cond: prepare(&cc)?,
            cross_kv_uncond: prepare(&cu)?,
            trunk,
        })
    }
}

/// The boundary-switched, APG-guided denoise loop (the Bernini analog of
/// `mlx_gen_wan::pipeline::denoise_moe`). Runs in **spatial latent space** `[16, T, H8, W8]`: each
/// step picks the high-noise expert while `t ≥ boundary` and the low-noise expert below it, multiplies
/// all omegas by `omega_scale` once on the first low-noise step, computes the per-mode guided velocity
/// (sigma = this step's flow sigma, for the APG x-conversion), and applies the UniPC flow step.
#[allow(clippy::too_many_arguments)]
fn denoise_bernini(
    pf: &PackedForward,
    mode: Mode,
    low: &BExpert,
    high: &BExpert,
    boundary: f32,
    num_train: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    videos: &[Array],
    images: &[Array],
    base_g: &GuidanceParams,
    omega_scale: f32,
    momentum: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(SolverKind::UniPC, num_train);
    sched.set_timesteps(steps, shift);
    let timesteps = sched.timesteps().to_vec();
    let sigmas = sched.sigmas().to_vec();

    let mut latent = init_noise.clone();
    let mut switched = false;
    let mut g = base_g.clone();
    let mut mbufs: Vec<MomentumBuffer> = (0..num_momentum_buffers(mode))
        .map(|_| MomentumBuffer::new(momentum))
        .collect();

    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract (F-003): a render runs minutes, so check before
        // each step. The per-step `eval` below makes this effective (lazy MLX graph otherwise defers
        // all compute past every check).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // 1-based, so the final step reaches `total` (the Progress::Step 1..=total contract, F-038).
        on_step(i + 1);
        let expert = if t >= boundary {
            high
        } else {
            if !switched {
                switched = true;
                g.omega_vid *= omega_scale;
                g.omega_img *= omega_scale;
                g.omega_txt *= omega_scale;
            }
            low
        };
        let sigma = sigmas[i];
        let v = guided_velocity(
            pf,
            mode,
            expert.trunk,
            &latent,
            videos,
            images,
            t,
            sigma,
            &expert.cross_kv_cond,
            &expert.cross_kv_uncond,
            &g,
            &mut mbufs,
        )?;
        latent = sched.step(&v, &latent)?;
        eval([&latent])?;
    }
    Ok(latent)
}

/// One expert (high or low) with its prepared cross-attention K/V for the planner's **4** prompt-embed
/// streams (the full-Bernini ViT-conditioned path). Each stream is `concat_with_zero_init(UMT5(prompt),
/// planner ViT-context)` (sc-5140), in renderer `text_dim` space, so it goes through the same
/// `embed_text` → `prepare_cross_kv` as the renderer's text context.
pub struct BVitExpert<'a> {
    trunk: Trunk<'a>,
    wtxt_wvit: Vec<(Array, Array)>,
    wtxt_wovit: Vec<(Array, Array)>,
    wotxt_wvit: Vec<(Array, Array)>,
    wotxt_wovit: Vec<(Array, Array)>,
}

impl<'a> BVitExpert<'a> {
    /// `streams` = `[wtxt_wvit, wtxt_wovit, wotxt_wvit, wotxt_wovit]` prompt-embed contexts.
    pub fn build(trunk: Trunk<'a>, streams: [&Array; 4]) -> Result<Self> {
        let dit = trunk.dit;
        // See `BExpert::build` for why a windowed trunk builds its cross-K/V through the plan.
        let prep = |s: &Array| -> Result<Vec<(Array, Array)>> {
            let embedded = dit.embed_text(s)?;
            match trunk.window {
                None => dit.prepare_cross_kv(&embedded),
                Some((stream, plan, cancel)) => {
                    stream.prepare_cross_kv_windowed(&embedded, plan, cancel)
                }
            }
        };
        Ok(Self {
            wtxt_wvit: prep(streams[0])?,
            wtxt_wovit: prep(streams[1])?,
            wotxt_wvit: prep(streams[2])?,
            wotxt_wovit: prep(streams[3])?,
            trunk,
        })
    }

    fn streams(&self) -> VitStreams<'_> {
        VitStreams {
            wtxt_wvit: &self.wtxt_wvit,
            wtxt_wovit: &self.wtxt_wovit,
            wotxt_wvit: &self.wotxt_wvit,
            wotxt_wovit: &self.wotxt_wovit,
        }
    }
}

/// The full-Bernini ViT-conditioned denoise loop (`sample_bernini_wvitcfg`, `wan_diffusion.py`
/// 571-793) — the renderer-side compute that consumes the planner's 4 prompt streams. The boundary-
/// switched, [`vit_one_step`]-guided analog of `denoise_bernini`: each step picks the expert by the
/// `switch_dit_boundary`, multiplies **all four** omegas (incl. `omega_tgt`) by `omega_scale` once on
/// the first low-noise step, and applies the UniPC flow step. Runs in spatial latent space
/// `[16, T, H8, W8]`. The full end-to-end string-up (planner MAR loop → these 4 streams → here →
/// decode) is the registered pipeline in sc-5145.
#[allow(clippy::too_many_arguments)]
pub fn denoise_bernini_wvitcfg(
    pf: &PackedForward,
    mode: VitMode,
    low: &BVitExpert,
    high: &BVitExpert,
    boundary: f32,
    num_train: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    images: &[(Array, f64)],
    videos: &[(Array, f64)],
    base_g: &VitGuidanceParams,
    omega_scale: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(SolverKind::UniPC, num_train);
    sched.set_timesteps(steps, shift);
    let timesteps = sched.timesteps().to_vec();
    let sigmas = sched.sigmas().to_vec();

    let mut latent = init_noise.clone();
    let mut switched = false;
    let mut g = base_g.clone();

    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract (F-003): check before each (minutes-long) step.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // 1-based, so the final step reaches `total` (the Progress::Step 1..=total contract, F-038).
        on_step(i + 1);
        let expert = if t >= boundary {
            high
        } else {
            if !switched {
                switched = true;
                g.omega_txt *= omega_scale;
                g.omega_img *= omega_scale;
                g.omega_vid *= omega_scale;
                g.omega_tgt *= omega_scale;
            }
            low
        };
        let v = vit_one_step(
            pf,
            expert.trunk,
            mode,
            &latent,
            images,
            videos,
            t,
            sigmas[i],
            &expert.streams(),
            &g,
        )?;
        latent = sched.step(&v, &latent)?;
        eval([&latent])?;
    }
    Ok(latent)
}

mlx_gen::impl_generator!(BerniniRenderer {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl BerniniRenderer {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // Shared geometry guard (sc-12454/F-003 + sc-12500/F-040): the sc-12308 14B video-only area
        // cap, the 16-px grid rejection (replacing the silent `align_dim` refit), and the
        // `1 + 4·k` frame rule — mirrored from candle's `validate_bernini_geometry` so the same
        // request gets the same rejection on both backends.
        validate_bernini_geometry(self.descriptor.id, req)?;
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let cfg = &self.config;
        let k = &self.knobs;

        // --- Resolve geometry + knobs ---
        let frames = req
            .frames
            .map(|f| f as usize)
            .unwrap_or(Defaults::NUM_FRAMES)
            .max(1);
        // sc-12500 (F-040): `validate_impl` rejects any off-grid width/height, so the reference's
        // align-down is a no-op here — assert that instead of silently refitting the request
        // (1000×1000 used to render 992×992 with no diagnostic while candle errored).
        let width = align_dim(req.width, cfg.patch_size.2, PROVIDER_VAE_STRIDE.2);
        let height = align_dim(req.height, cfg.patch_size.1, PROVIDER_VAE_STRIDE.1);
        debug_assert_eq!(
            (width, height),
            (req.width, req.height),
            "bernini_renderer: off-grid request survived validate"
        );
        let steps = req.steps.map(|s| s as usize).unwrap_or(Defaults::STEPS);
        let seed = req.seed.unwrap_or(42);
        let neg = req.negative_prompt.clone().unwrap_or_default();

        // Conditioning split: VideoClip → videos, Reference/MultiReference → images.
        let has_video = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::VideoClip { .. }));
        let has_image = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
            )
        });
        let mode = resolve_mode(req.video_mode.as_deref(), has_video, has_image);

        let omega_txt = req.guidance.unwrap_or(Defaults::OMEGA_TXT);
        let base_g = GuidanceParams {
            omega_vid: Defaults::OMEGA_VID,
            omega_img: Defaults::OMEGA_IMG,
            omega_txt,
            eta: Defaults::ETA,
            norm_threshold: [Defaults::NORM_THRESHOLD, Defaults::NORM_THRESHOLD],
        };

        let lat = latent_shape(frames, height, width, cfg.vae_z_dim, PROVIDER_VAE_STRIDE)?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = enc.encode(&tokenizer, &neg)?;
            eval([&context, &context_null])?;
            (context, context_null)
        };
        // Residency discipline (epic 10834, sc-10840): the UMT5-XXL text encoder + its `Weights` are
        // dropped at the block close above and `context`/`context_null` are materialized — flush the
        // freed encoder pages back to the OS NOW, before the source-VAE encode and the two ~28 GB expert
        // loads, so they do not linger in MLX's buffer cache and inflate the resident set. The live
        // conditioning arrays are untouched, so this is byte-identical (a pure memory hint). Mirrors the
        // `bernini` id's two encoder-drop flushes (bernini.rs).
        clear_cache();
        // F-135: T5 encode materialized above; honor a cancel before the source VAE encodes and the
        // two ~28 GB expert loads (all pre-denoise stages sc-9093's loop-only cancel didn't reach).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- Stage 1b: VAE-encode source media → conditioning latents (→ encoder freed) ---
        let (videos, images) = if has_video || has_image {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = ProviderVae::from_weights(&w)?;
            let mut videos = Vec::new();
            let mut images = Vec::new();
            for c in &req.conditioning {
                match c {
                    Conditioning::VideoClip { frames, .. } => {
                        videos.push(encode_videoclip(&vae, frames, width, height)?)
                    }
                    Conditioning::Reference { image, .. } => {
                        images.push(encode_image(&vae, image, width, height)?)
                    }
                    Conditioning::MultiReference { images: imgs } => {
                        for im in imgs {
                            images.push(encode_image(&vae, im, width, height)?);
                        }
                    }
                    _ => {}
                }
            }
            let all: Vec<&Array> = videos.iter().chain(images.iter()).collect();
            if !all.is_empty() {
                eval(all)?;
            }
            (videos, images)
        } else {
            (Vec::new(), Vec::new())
        };
        // Residency discipline (sc-10840): the Stage-1b source-VAE encoder + its `Weights` are dropped at
        // the block close above and the `videos`/`images` conditioning latents are materialized — flush
        // the freed VAE-encoder pages before the two ~28 GB expert loads (a fresh VAE is reloaded for the
        // Stage-3 decode). Live latents untouched, byte-identical — the second of the two encoder-drop
        // boundaries, matching the `bernini` id.
        clear_cache();
        // F-135: source encodes materialized (if any); honor a cancel before the expert loads.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // Seeded init noise (spatial, f32). Bit-parity vs torch needs the reference's CPU-MT19937
        // draw injected; the coherence bar uses the MLX RNG.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 2: load both experts, APG denoise (→ experts freed) ---
        // Load+quantize each expert before loading the next so only one bf16 transient is resident at
        // a time (sc-5360 — `WanTransformer::quantize` eval-frees its bf16 dequant). Without quant this
        // just loads both bf16.
        // ── Ladder rungs 3 + 4 (sc-15528) ────────────────────────────────────────────────────
        // Rung 4 is per-request: when a window is selected the expert's block stack is NEVER
        // materialized (`from_weights_deferred` holds zero blocks) and the stream rebuilds
        // `plan.window()` blocks at a time. On a DUAL-expert config that is the whole point --
        // sc-16354's finding is that a naive per-expert window leaves the idle expert's 40 blocks
        // resident, and a stack that was never materialized has no idle half to pay for. Both
        // experts are deferred here, so the bound is over the full 80-block trunk, not one expert.
        let window = crate::memory_strategy::transformer_window_size(req)?;
        let attn_budget = crate::memory_strategy::attention_budget(req);
        let plan = window
            .map(|size| BlockPlan::new(cfg.num_layers, size))
            .transpose()?;
        let load_expert = |name: &str| -> Result<(WanTransformer, Option<WanBlockStream>)> {
            let path = self.root.join(name);
            let w = Weights::from_file(&path)?;
            if window.is_some() {
                let dit = WanTransformer::from_weights_deferred(&w, cfg)?;
                // No adapters can reach here (`supports_lora: false`), and the stream refuses an
                // adapted load anyway -- Wan MERGES deltas at load, so a streamed block re-read from
                // the snapshot would silently carry none of them.
                let mut stream = WanBlockStream::new(WeightsSource::File(path), cfg.clone(), &[])?;
                if let Some(q) = self.quant {
                    // Replayed per materialized block so the streamed weights are byte-identical to
                    // the resident ones. A pre-packed tier takes this through `cfg.quantization`
                    // inside the block constructor instead, exactly as the resident path does.
                    stream.set_quant_bits(q.bits());
                }
                stream.set_attention_budget(attn_budget);
                Ok((dit, Some(stream)))
            } else {
                let mut dit = WanTransformer::from_weights(&w, cfg)?;
                if let Some(q) = self.quant {
                    dit.quantize(q.bits(), None)?;
                }
                dit.set_attention_budget(attn_budget);
                Ok((dit, None))
            }
        };
        let latents = {
            let (low_dit, low_stream) = load_expert("low_noise_model.safetensors")?;
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let (high_dit, high_stream) = load_expert("high_noise_model.safetensors")?;
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let low = BExpert::build(
                Trunk::for_load(&low_dit, low_stream.as_ref(), plan.as_ref(), &req.cancel),
                &context,
                &context_null,
            )?;
            let high = BExpert::build(
                Trunk::for_load(&high_dit, high_stream.as_ref(), plan.as_ref(), &req.cancel),
                &context,
                &context_null,
            )?;
            let pf = PackedForward::new(
                cfg.dim / cfg.num_heads,
                cfg.out_dim,
                cfg.patch_size,
                k.max_trained_src_id,
                k.interpolate_src_id,
            );
            let boundary = k.switch_dit_boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            // `denoise_bernini` reports 1-based steps, so `current` runs exactly 1..=total (F-038).
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_bernini(
                &pf,
                mode,
                &low,
                &high,
                boundary,
                cfg.num_train_timesteps,
                steps,
                k.shift,
                &init_noise,
                &videos,
                &images,
                &base_g,
                Defaults::OMEGA_SCALE,
                Defaults::MOMENTUM,
                &req.cancel,
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        let (out_frames, out_height, out_width) = decoded_output_geometry(lat[1], lat[2], lat[3])?;
        // Ladder rung 2 (sc-15528) — see the note on the full pipeline's decode.
        let tiling = match crate::memory_strategy::decode_tiling(req)? {
            Some(explicit) => Some(explicit),
            None => TilingConfig::auto(out_height, out_width, out_frames),
        };
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = ProviderVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref(), Some(&req.cancel))?
        };
        let images_out = frames_to_images(&frames_u8)?;

        // num_frames == 1 ⇒ a still image (t2i/i2i). A single latent frame (T_lat = 1) still decodes
        // to one VAE temporal chunk (the causal decode emits `vae_stride_t` near-identical frames);
        // the still image is the first of them, matching the reference's single-frame PNG.
        if frames == 1 {
            let first = images_out.into_iter().next().ok_or_else(|| {
                Error::Msg("bernini_renderer: VAE decode produced no frames".into())
            })?;
            Ok(GenerationOutput::Images(vec![first]))
        } else {
            let fps = req.fps.unwrap_or(16);
            Ok(GenerationOutput::Video {
                frames: images_out,
                fps,
                audio: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::multiply;

    /// Component residency (epic 10834, sc-10840): the renderer is structurally always-staged —
    /// `generate_impl` drops the UMT5 text encoder and the
    /// source-VAE encoder (each + `clear_cache()`) before loading the two co-resident MoE experts, so
    /// peak unified memory is already bounded to the dominant expert phase (the footprint's DiT split).
    #[test]
    fn declares_unconditional_staged_residency_without_a_selectable_control() {
        let capabilities = descriptor().capabilities;
        assert!(!capabilities.supports_sequential_offload);
        assert!(capabilities.unconditionally_engages_staged_residency);
        assert_eq!(
            capabilities.staged_residency_availability(),
            mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged,
        );
    }

    fn tiny_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_1_3b();
        c.dim = 128;
        c.num_heads = 1;
        c.num_layers = 2;
        c.ffn_dim = 256;
        c.freq_dim = 256;
        c.text_dim = 32;
        c.text_len = 8;
        c.in_dim = 16;
        c.out_dim = 16;
        c.vae_z_dim = 16;
        c.boundary = 0.875;
        c.num_train_timesteps = 1000;
        c
    }

    fn load(name: &str) -> Weights {
        let path = format!(
            "{}/../mlx-gen-wan/tests/fixtures/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        Weights::from_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// The ViT-conditioned denoise loop runs end-to-end over a tiny dual-expert (crossing the
    /// boundary so the `omega_scale` switch fires) and preserves the spatial latent shape. Pins the
    /// loop plumbing (scheduler / expert switch / per-step `vit_one_step` / flow step); the per-step
    /// math is validated in slices A/B and the e2e coherence in sc-5145.
    #[test]
    fn denoise_wvitcfg_runs_and_keeps_shape() {
        let w = load("s5_low.safetensors");
        let cfg = tiny_cfg();
        let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");
        let pf = PackedForward::new(
            cfg.dim / cfg.num_heads,
            cfg.out_dim,
            cfg.patch_size,
            5.0,
            true,
        );
        let noisy = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
        let cc = w.require("ctx_cond").unwrap();
        let cu = w.require("ctx_uncond").unwrap();
        let scale = |a: &Array, s: f32| multiply(a, Array::from_f32(s)).unwrap();
        // 4 distinct prompt streams.
        let (s0, s1, s2, s3) = (
            scale(cc, 1.0),
            scale(cu, 1.0),
            scale(cc, 0.5),
            scale(cu, 0.5),
        );
        let streams = [&s0, &s1, &s2, &s3];
        let low = BVitExpert::build(Trunk::resident(&dit), streams).expect("low expert");
        let high = BVitExpert::build(Trunk::resident(&dit), streams).expect("high expert");
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 3.0,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let mut on_step = |_i: usize| {};
        let out = denoise_bernini_wvitcfg(
            &pf,
            VitMode::VaeTxtVitWapg,
            &low,
            &high,
            0.875 * cfg.num_train_timesteps as f32, // boundary 875 → crosses with 3 steps
            cfg.num_train_timesteps,
            3,
            5.0,
            noisy,
            &[],
            &[],
            &g,
            0.8,
            &CancelFlag::default(),
            &mut on_step,
        )
        .expect("denoise");
        assert_eq!(
            out.shape(),
            noisy.shape(),
            "loop preserves spatial latent shape"
        );
    }
}
