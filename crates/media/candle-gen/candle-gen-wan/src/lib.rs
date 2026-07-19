//! # candle-gen-wan
//!
//! The **Wan2.2 TI2V-5B** text-to-video provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-wan`. Wan has **no** `candle-transformers` reference: the
//! `WanTransformer3DModel` DiT ([`transformer`]), the causal-Conv3d `AutoencoderKLWan` temporal VAE
//! ([`vae`], built on a from-scratch [`conv3d`] since candle ships none), the UMT5-XXL encoder
//! ([`text_encoder`]), and the UniPC flow-match scheduler ([`scheduler`]) are all ported here from
//! the diffusers checkpoint.
//!
//! **txt2video (sc-3697):** [`WanGenerator::generate`] runs UMT5-XXL → the 30-layer DiT (3-axis
//! interleaved RoPE, AdaLN modulation, cross-attention to text, classifier-free guidance, UniPC) →
//! the temporal VAE decoder, emitting `GenerationOutput::Video`. Registered under `"wan2_2_ti2v_5b"`.
//!
//! **Dtypes:** the 5B DiT runs **bf16** (its native dtype), norms/modulation upcast to f32; the UMT5
//! encoder runs **bf16** (sc-12778 — halving the f32 encoder resident + its ~24 GB ENCODE-stage
//! transient, the 5B sequential-offload <16 GB lever, epic sc-12732; the DiT `embed_text` already
//! casts the context to bf16, so this REMOVES the old f32→bf16 boundary); the VAE runs **f32**.
//! `backend = "candle"`, `mac_only = false`.
//!
//! **First-slice surface:** txt2video only. The mlx provider's image-conditioning (TI2V / I2V),
//! VACE, LoRA, and quantization surface is **deferred**. The z48 vae22 decode is memory-bounded:
//! the temporal axis streams per-frame ([`vae::WanVae::decode`]) and a budgeted **spatial** tiler
//! ([`vae::WanVae::decode_budgeted`], sc-7111) caps a single high-res frame's VRAM spike.

pub mod adapters;
pub mod candle_tier_build;
// ComfyUI single-file Wan2.2 expert → in-memory remap+dequant seam (epic 10451 Phase 2c, sc-10671):
// scaled-fp8 dequant (`w = w_fp8·scale_weight`) + native-Wan → diffusers key remap, so a user's existing
// ComfyUI Wan base experts load in place via `VarBuilder::from_tensors`. Entry: `load_from_comfyui_experts`.
mod comfyui;
pub mod config;
pub mod conv3d;
pub mod dit_train;
// Native GGUF k-quant DiT loader (sc-12735, epic 12732 Pillar 2 — the 24 GB lever): opens a
// `QuantStack/Wan2.2-TI2V-5B-GGUF` `.gguf` and holds the DiT as resident Q4_K_M `QTensor`s that
// dequantize per-matmul (ComfyUI-GGUF parity), NEVER pre-dequantized to dense at load. Selected on the
// 5B by the `CANDLE_GEN_WAN_GGUF` sub-story-1 test seam (manifest/catalog routing is sub-story 2).
mod gguf;
pub mod model_vace;
pub mod pipeline;
pub mod quant;
pub mod rope;
pub mod scheduler;
mod text_encode;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vace;
pub mod vae;
pub mod vae16;
pub mod wan14b;

/// Operational Wan video ceiling: `1 + 4 * 256` pixel frames.
pub(crate) const MAX_WAN_FRAMES: usize = 1025;
/// Matching temporal-conditioning budget after the z16 VAE's 4x causal compression.
pub(crate) const MAX_WAN_CONDITIONING_LATENTS: usize = 257;

pub(crate) fn combined_conditioning_latents(
    control_frames: usize,
    reference_images: usize,
) -> Option<usize> {
    let control_latents = control_frames
        .checked_sub(1)?
        .checked_div(4)?
        .checked_add(1)?;
    control_latents.checked_add(reference_images)
}

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::{CancelFlag, LoadPhase};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, MoeExpert, OffloadPolicy, Progress, Quant, WeightsSource,
};
use candle_gen::{check_cancel, effective_offload_policy, CandleError, Result as CResult};

use candle_gen::gen_core::sampling::TimestepConvention;
use config::{
    TextEncoderConfig, TransformerConfig, VaeConfig, DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_GUIDANCE,
    DEFAULT_STEPS, MAX_AREA_5B, MIN_SIZE, MODEL_ID, NEGATIVE_FALLBACK, SIZE_MULTIPLE,
};
use rope::WanRope;
use scheduler::{flow_shift, FlowScheduler, Sampler};
use text_encoder::Umt5Encoder;
use transformer::WanTransformer;
use vae::WanVae;

/// The 5B DiT runs bf16 (native checkpoint dtype); the UMT5 encoder runs bf16 (sc-12778 — halving the
/// f32 encoder's ~24 GB ENCODE-stage transient to ~12 GB, the 5B sequential <16 GB lever, epic
/// sc-12732; the DiT's `embed_text` already casts the context to bf16, so this REMOVES the old
/// f32→bf16 boundary rather than adding one); the VAE runs f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;
const Z_DIM: usize = 48;

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    dit: Arc<WanTransformer>,
    vae: Arc<WanVae>,
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across every prompt/branch
    /// encode (sc-8991 / F-011) rather than re-parsing `tokenizer.json` per request.
    tok: Arc<candle_gen::gen_core::tokenizer::TextTokenizer>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    vae_cfg: VaeConfig,
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters to apply to the DiT at load (sc-10095). On a dense tier they FOLD into the
    /// weights ([`adapters::merge_adapters`]); on a packed q4/q8 tier they attach as forward-time
    /// **additive** residuals ([`adapters::install_additive`], sc-10094) — a packed tier has no dense
    /// `W` to fold into.
    adapters: Vec<AdapterSpec>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, adapters: Vec<AdapterSpec>) -> Self {
        Self {
            adapters,
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: TransformerConfig::ti2v_5b(),
            vae_cfg: VaeConfig::ti2v_5b(),
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        // Shared Wan component loader (sc-9000 / F-020); the crafted snapshot description stays local.
        text_encode::component_vb(
            &self.root,
            sub,
            dtype,
            &self.device,
            "wan",
            "Wan2.2-TI2V-5B diffusers",
        )
    }

    fn load_components(&self) -> CResult<Components> {
        let te = self.load_te()?;
        let dit = self.build_dit()?;
        let vae = self.load_vae()?;
        let tok = text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan")?;
        Ok(Components {
            te: Arc::new(te),
            dit: Arc::new(dit),
            vae: Arc::new(vae),
            tok: Arc::new(tok),
        })
    }

    /// Build the UMT5 text encoder (bf16, sc-12778) — the ~11 GB phase-A component (was ~21 GB f32) that
    /// is dead weight after the prompt encode; bf16 also halves its ~24 GB f32 ENCODE-stage transient
    /// (~12 GB), the residual peak sc-12757 measured. A single home shared by the resident
    /// [`load_components`](Self::load_components) build and the sequential-offload
    /// [`render_sequential`](Self::render_sequential) stage, so the two paths can never diverge in how
    /// the TE is built (sc-12757).
    fn load_te(&self) -> CResult<Umt5Encoder> {
        Ok(Umt5Encoder::new(
            &self.te_cfg,
            self.component_vb("text_encoder", ENC_DTYPE)?,
        )?)
    }

    /// Build the z48 vae22 VAE (f32) — the decode-stage component. Shared by the resident and staged
    /// paths so the residency change stays a residency change only (sc-12757).
    fn load_vae(&self) -> CResult<WanVae> {
        Ok(WanVae::new(
            &self.vae_cfg,
            self.component_vb("vae", VAE_DTYPE)?,
        )?)
    }

    /// Build the TI2V-5B DiT, applying [`Self::adapters`] by tier (sc-10095): a **dense** tier folds the
    /// delta into the weights ([`adapters::merge_adapters`], the merge-not-residual fast path, byte
    /// identical to before); a **packed** q4/q8 tier attaches forward-time **additive** residuals on the
    /// packed `QLinear` ([`adapters::install_additive`], sc-10094) — a packed tier has no dense `W` to
    /// fold into, and LoKr/LoHa on it is rejected there (deferred to sc-10050/10051). The 5B is a single
    /// (non-MoE) DiT, so every adapter is shared (`moe_expert = None`); the `expert` arg is a formality.
    fn build_dit(&self) -> CResult<WanTransformer> {
        // sub-story-1 test seam (sc-12735): a native-GGUF k-quant DiT path, selected by the
        // `CANDLE_GEN_WAN_GGUF` env var pointing at a downloaded `QuantStack/Wan2.2-TI2V-5B-GGUF` `.gguf`.
        // The DiT is held as resident Q4_K_M `QTensor`s (dequant-on-matmul) — the loader-proof this PR
        // lands. Manifest/catalog/tier routing is sub-story 2; adapter routing on this path is a later
        // sub-story, so a LoRA/LoKr spec on the GGUF seam is rejected loudly rather than silently ignored.
        if let Some(gguf) = crate::gguf::env_gguf_path() {
            if !self.adapters.is_empty() {
                return Err(CandleError::Msg(format!(
                    "wan: LoRA/LoKr on the native-GGUF 5B path ({}) is not wired yet — sc-12735 sub-story \
                     1 is the GGUF loader mechanism; adapter routing on the GGUF tier is a later sub-story",
                    crate::gguf::GGUF_ENV
                )));
            }
            // candle_core::Result → CResult (CandleError) via the `?` bridge.
            return Ok(crate::gguf::load_wan_dit_gguf(
                &gguf,
                &self.dit_cfg,
                &self.device,
                DIT_DTYPE,
            )?);
        }
        let vb = self.component_vb("transformer", DIT_DTYPE)?;
        // Packed-tier marker: the sc-10025 seam packs every DiT Linear (incl. `proj_out`).
        let packed = vb.contains_tensor("proj_out.scales");
        if packed {
            let mut dit = WanTransformer::new(&self.dit_cfg, vb)?;
            if self.adapters.is_empty() {
                return Ok(dit);
            }
            let report = adapters::install_additive(&mut dit, &self.adapters, MoeExpert::High)?;
            if report.applied == 0 {
                return Err(CandleError::Msg(format!(
                    "wan: {} LoRA adapter file(s) matched no projection on the packed TI2V-5B DiT — \
                     check the key format (expected PEFT `<path>.lora_A/B.weight` or kohya \
                     `lora_unet_<flat>` targeting the DiT attention/FFN Linears)",
                    self.adapters.len()
                )));
            }
            return Ok(dit);
        }
        if self.adapters.is_empty() {
            return Ok(WanTransformer::new(&self.dit_cfg, vb)?);
        }
        // Dense tier + adapters: fold the delta into the dense weights before build (`merge_adapters`
        // hard-errors on its own zero-match).
        drop(vb);
        let mut map = text_encode::load_component_map(&self.root, "transformer", "wan")?;
        adapters::merge_adapters(&mut map, &self.adapters)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok(WanTransformer::new(&self.dit_cfg, vb)?)
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (bf16, zero-padded to `max_length`). Shared
    /// Wan text-encode routine (sc-9000 / F-020); ENC_DTYPE is bf16 (sc-12778) so the encoder resident +
    /// its ENCODE-stage transient halve, and the DiT `embed_text` bf16 cast is now a no-op.
    fn encode(&self, comps: &Components, prompt: &str) -> CResult<Tensor> {
        self.encode_raw(&comps.tok, &comps.te, prompt)
    }

    /// The tokenizer+encoder core of [`encode`](Self::encode), taking the two text components directly
    /// so the sequential path can drive it with its staged (about-to-be-dropped) UMT5 encoder rather
    /// than the resident [`Components`] bundle (sc-12757). Produces the same raw `[1, 512, 4096]` f32
    /// context either way — the DiT's `embed_text` projection happens later, once the DiT loads.
    fn encode_raw(&self, tok: &TextTokenizer, te: &Umt5Encoder, prompt: &str) -> CResult<Tensor> {
        text_encode::umt5_encode_padded(
            tok,
            &self.te_cfg,
            te,
            prompt,
            &self.device,
            ENC_DTYPE,
            "wan",
        )
    }

    /// Resolve the per-request knobs against the config defaults — shared verbatim by the resident
    /// [`render`](Self::render) and the sequential [`render_sequential`](Self::render_sequential) so the
    /// two paths can never resolve steps/guidance/shift differently (the residency change must be
    /// numerics-preserving, sc-12757).
    fn resolve_knobs(&self, req: &GenerationRequest) -> RenderKnobs {
        RenderKnobs {
            steps: req
                .steps
                .map(|s| s as usize)
                .unwrap_or(DEFAULT_STEPS as usize),
            frames: req.frames.unwrap_or(DEFAULT_FRAMES),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE) as f64,
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
            sampler: Sampler::parse(req.sampler.as_deref()),
            shift: flow_shift(req.scheduler_shift),
        }
    }

    /// Latent geometry (z48 strides) + the token-grid RoPE tables `(t_lat, h_lat, w_lat, cos, sin)`.
    /// Shared by both render paths so they seed the identical noise grid + RoPE (sc-12757).
    fn geometry(
        &self,
        req: &GenerationRequest,
        frames: u32,
    ) -> CResult<(usize, usize, usize, Tensor, Tensor)> {
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(frames, req.width, req.height);
        let (pt, ph, pw) = self.dit_cfg.patch;
        let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.dit_cfg).cos_sin(ppf, pph, ppw, &self.device)?;
        Ok((t_lat, h_lat, w_lat, cos, sin))
    }

    /// Run the whole denoise from `latents0` to the final latent on the resident (single, dense) DiT,
    /// returning the denoised latent. Extracted verbatim out of the monolithic render so the resident
    /// and sequential paths drive the **identical** per-step math — the epic 7114 P4 curated fold-in
    /// branch and the byte-exact native-`FlowScheduler` branch both (sc-12757, residency change only).
    ///
    /// epic 7114 P4 (sc-7124) Wan fold-in: the gen-core-only curated solvers (euler_ancestral / heun /
    /// dpmpp_sde / ddim) run over Wan's NATIVE flow σ schedule via the shared driver — one solver
    /// library. Wan's native UniPC (curated `uni_pc`, sc-7296) / `euler` (the diffusers FLOW-SNR
    /// multistep + flow Euler) stay the byte-exact default path; gen-core's VE-space `uni_pc`/`dpmpp_2m`
    /// are deliberately NOT routed through the fold-in (they would diverge from Wan's diffusers parity).
    /// The DiT timestep is `σ·N` (Sigma convention, ×N applied in the closure); the model output is the
    /// velocity (CFG combined inside).
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        dit: &WanTransformer,
        ctx_pos: &Tensor,
        ctx_neg: Option<&Tensor>,
        guidance: f64,
        cos: &Tensor,
        sin: &Tensor,
        latents0: Tensor,
        knobs: &RenderKnobs,
        sampler_name: Option<&str>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Tensor> {
        let steps = knobs.steps;
        let shift = knobs.shift;
        const FOLDIN: &[&str] = &["euler_ancestral", "heun", "dpmpp_sde", "ddim"];
        let latents = if let Some(name) = sampler_name.filter(|n| FOLDIN.contains(n)) {
            let native = scheduler::flow_sigmas(steps, shift);
            let n_train = config::NUM_TRAIN_TIMESTEPS as f64;
            candle_gen::run_flow_sampler(
                Some(name),
                TimestepConvention::Sigma,
                &native,
                latents0,
                knobs.seed,
                cancel,
                on_progress,
                |latents, t| -> CResult<Tensor> {
                    let ts = t as f64 * n_train;
                    let v_pos = dit.forward(latents, ctx_pos, ts, cos, sin)?;
                    let v = match ctx_neg {
                        Some(neg) => {
                            let v_neg = dit.forward(latents, neg, ts, cos, sin)?;
                            pipeline::cfg(&v_pos, &v_neg, guidance)?
                        }
                        None => v_pos,
                    };
                    Ok(v)
                },
            )?
        } else {
            // Native FlowScheduler (UniPC default / flow Euler) — the byte-exact N1 path, untouched.
            let mut latents = latents0;
            let mut sched = FlowScheduler::new(knobs.sampler, steps, shift);
            let total = steps as u32;
            for i in 0..steps {
                check_cancel(cancel)?;
                let t = sched.timestep(i);
                let v_pos = dit.forward(&latents, ctx_pos, t, cos, sin)?;
                let v = match ctx_neg {
                    Some(neg) => {
                        let v_neg = dit.forward(&latents, neg, t, cos, sin)?;
                        pipeline::cfg(&v_pos, &v_neg, guidance)?
                    }
                    None => v_pos,
                };
                latents = sched.step(&v, &latents)?;
                on_progress(Progress::Step {
                    current: i as u32 + 1,
                    total,
                });
            }
            latents
        };
        Ok(latents)
    }

    /// The resident render (unchanged residency: `Components` holds UMT5, the DiT and the VAE
    /// co-resident for the whole generation). Encodes once, projects the DiT context, denoises, decodes.
    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let knobs = self.resolve_knobs(req);

        // Text encode (pos + optional neg for CFG), then project to the DiT context once.
        let pos_embeds = self.encode(comps, &req.prompt)?;
        let ctx_pos = comps.dit.embed_text(&pos_embeds)?;
        let ctx_neg = if knobs.guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
            Some(comps.dit.embed_text(&self.encode(comps, neg)?)?)
        } else {
            None
        };

        let (t_lat, h_lat, w_lat, cos, sin) = self.geometry(req, knobs.frames)?;
        let latents0 =
            pipeline::create_noise(knobs.seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;

        let latents = self.denoise(
            &comps.dit,
            &ctx_pos,
            ctx_neg.as_ref(),
            knobs.guidance,
            &cos,
            &sin,
            latents0,
            &knobs,
            req.sampler.as_deref(),
            &req.cancel,
            on_progress,
        )?;

        on_progress(Progress::Decoding);
        // Memory-bounded z48 vae22 decode (sc-7111): the per-frame streaming `decode` already bounds
        // the temporal axis; `decode_budgeted` adds budgeted **spatial** tiling so a single high-res
        // frame can't spike VRAM, and returns a catchable error rather than OOM-ing when over budget.
        let decoded = comps
            .vae
            .decode_budgeted_with_cancel(&latents, &req.cancel)?;
        let images = pipeline::frames_to_images(&decoded)?;
        Ok((images, knobs.fps))
    }

    /// The **sequential-offload** render (sc-12757, epic 12732) — the staged twin of
    /// [`render`](Self::render) that keeps only **one** heavy component GPU-resident at a time, so the
    /// bf16 UMT5 encoder (~11 GB, dead weight after the encode) is off-GPU for the entire denoise and the
    /// DiT is freed before the VAE loads. sc-12757 measured the residual sequential peak at the UMT5
    /// **ENCODE-stage transient** (the f32 encoder's ~24 GB, in stage 1 before the drop — NOT the denoise
    /// or decode), so sc-12778 runs the encoder in bf16 to halve that transient (~12 GB), landing the 5B
    /// sequential peak under the epic's 16 GB target. The 5B is a single dense DiT (no MoE / expert swap),
    /// so the stages are linear:
    ///
    /// 1. **TE off-GPU during denoise.** Load UMT5, encode the pos (+ neg when CFG) **raw** `[1,512,4096]`
    ///    context, then DROP the ~11 GB bf16 encoder — only the small raw context tensors survive.
    /// 2. **DiT only for the denoise.** Load the DiT, project the raw context through its own
    ///    `embed_text`, run the full denoise, then DROP the DiT before the VAE materializes.
    /// 3. **VAE decode.** Load the VAE and `decode_budgeted`.
    ///
    /// Parity: the DiT `embed_text` projection is DiT-entangled, so it must happen after the DiT loads —
    /// the raw UMT5 context stays resident across the TE drop. The denoise runs the identical
    /// [`denoise`](Self::denoise) helper as the resident path, so the residency change is numerics-only.
    /// Each heavy component is a local bound to its own scope (driven through [`staged_sequential`]), so
    /// Rust's scope drop frees it before the next loads; a `device.synchronize()` at each boundary drains
    /// the async encode/denoise kernels before the freed pool is reused (the sc-12195 eviction race).
    fn render_sequential(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let knobs = self.resolve_knobs(req);
        let cancel = &req.cancel;
        check_cancel(cancel)?;

        // The tiny tokenizer is cheap and stays resident across the whole render; the heavy bf16 UMT5
        // encoder (~11 GB, sc-12778) is dropped right after encoding.
        let tok = text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan")?;
        let (t_lat, h_lat, w_lat, cos, sin) = self.geometry(req, knobs.frames)?;
        let latents0 =
            pipeline::create_noise(knobs.seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;

        // The cross-stage tensors that must survive a component drop: the raw UMT5 context (stage 1 →
        // stage 2) and the denoised latents (seeded here, denoised in stage 2, decoded in stage 3).
        let mut state = SeqState {
            pos: None,
            neg: None,
            latents: Some(latents0),
            on_progress: &mut *on_progress,
        };

        staged_sequential(
            &mut state,
            // ── Stage 1: load UMT5 → raw pos/neg context. DROP the ~11 GB bf16 encoder at the brace. ──
            |st| {
                check_cancel(cancel)?;
                (st.on_progress)(Progress::Loading(LoadPhase::TextEncoder));
                self.load_te()
            },
            |te, st| {
                st.pos = Some(self.encode_raw(&tok, te, &req.prompt)?);
                st.neg = if knobs.guidance > 1.0 {
                    let neg_prompt = req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK);
                    Some(self.encode_raw(&tok, te, neg_prompt)?)
                } else {
                    None
                };
                Ok(())
            },
            // ── Stage 2: load the DiT → project + denoise. DROP the DiT before the VAE loads. ──
            |st| {
                check_cancel(cancel)?;
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                self.build_dit()
            },
            |dit, st| {
                let pos = st.pos.as_ref().expect("pos context encoded in stage 1");
                let ctx_pos = dit.embed_text(pos)?;
                let ctx_neg = match &st.neg {
                    Some(neg) => Some(dit.embed_text(neg)?),
                    None => None,
                };
                let latents0 = st.latents.take().expect("noise seeded before staging");
                let latents = self.denoise(
                    dit,
                    &ctx_pos,
                    ctx_neg.as_ref(),
                    knobs.guidance,
                    &cos,
                    &sin,
                    latents0,
                    &knobs,
                    req.sampler.as_deref(),
                    cancel,
                    st.on_progress,
                )?;
                st.latents = Some(latents);
                Ok(())
            },
            // ── Stage 3: load the VAE → decode. ──
            |st| {
                check_cancel(cancel)?;
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                self.load_vae()
            },
            |vae, st| {
                (st.on_progress)(Progress::Decoding);
                let latents = st.latents.as_ref().expect("latents denoised in stage 2");
                let decoded = vae.decode_budgeted_with_cancel(latents, cancel)?;
                let images = pipeline::frames_to_images(&decoded)?;
                Ok((images, knobs.fps))
            },
            // sc-12195 boundary sync: drain kernels before the used component's pool is reused.
            || Ok(self.device.synchronize()?),
        )
    }
}

/// Resolved per-request render knobs (steps/guidance/shift/…), produced by
/// [`Pipeline::resolve_knobs`] and consumed identically by the resident and sequential render paths so
/// the residency change stays numerics-preserving (sc-12757).
struct RenderKnobs {
    steps: usize,
    frames: u32,
    fps: u32,
    guidance: f64,
    seed: u64,
    sampler: Sampler,
    shift: f64,
}

/// The mutable render state threaded through [`staged_sequential`] (sc-12757): the cross-stage tensors
/// that must outlive a component drop (the raw UMT5 context; the latents seeded → denoised → decoded)
/// plus the progress sink. Held so exclusive access moves between the load/use closures via the
/// `&mut SeqState` param rather than being captured by each closure — the borrow-checker-clean way to
/// let the `FnOnce` stages share state without a `RefCell` (mirrors `wan14b`'s `SwapState`).
struct SeqState<'a> {
    /// Raw UMT5 positive context, encoded in stage 1 (TE resident), projected + consumed in stage 2
    /// (DiT resident) — survives the TE drop.
    pos: Option<Tensor>,
    /// Raw UMT5 negative context (CFG only, else `None`), same lifetime as [`Self::pos`].
    neg: Option<Tensor>,
    /// Working latents: seeded before staging, denoised in stage 2 (survives the TE drop), decoded in
    /// stage 3 (survives the DiT drop).
    latents: Option<Tensor>,
    on_progress: &'a mut dyn FnMut(Progress),
}

/// Drive the dense TI2V-5B sequential offload so at most **one** heavy component is GPU-resident at a
/// time (sc-12757, the dense Pillar-1 win — no expert swap, the 5B is a single DiT): the ~11 GB bf16
/// UMT5 text encoder (sc-12778), then the bf16 DiT, then the f32 VAE. Each component is a local bound to its own
/// block, so Rust's scope drop frees it **before** the next loads — the TE drops before the DiT loads
/// (so it is off-GPU for the whole denoise) and the DiT drops before the VAE loads. `sync` runs at each
/// stage boundary — after the component is used and before it drops (and before the next loads) — so
/// in-flight kernels are drained before the freed allocator pool is reused (the sc-12195 eviction race).
///
/// Generic over the component types and the threaded state `St` so a CPU unit test can pin the
/// never-co-resident + drop-order properties with a lightweight liveness witness — no GPU, no real
/// weights — exactly as `wan14b`'s `staged_expert_swap` is pinned. The load closures receive `&mut St`
/// so they can emit their [`Progress::Loading`] before the (heavy) load; the use closures receive
/// `&mut St` to read/advance the cross-stage tensors.
#[allow(clippy::too_many_arguments)]
fn staged_sequential<Te, Dit, Vae, St, R>(
    state: &mut St,
    load_te: impl FnOnce(&mut St) -> CResult<Te>,
    use_te: impl FnOnce(&Te, &mut St) -> CResult<()>,
    load_dit: impl FnOnce(&mut St) -> CResult<Dit>,
    use_dit: impl FnOnce(&Dit, &mut St) -> CResult<()>,
    load_vae: impl FnOnce(&mut St) -> CResult<Vae>,
    use_vae: impl FnOnce(&Vae, &mut St) -> CResult<R>,
    mut sync: impl FnMut() -> CResult<()>,
) -> CResult<R> {
    // Stage 1: the UMT5 text encoder is resident ONLY for the encode.
    {
        let te = load_te(state)?;
        use_te(&te, state)?;
        // Drain the encode before `te` frees at the brace below and the DiT reuses the pool.
        sync()?;
    } // `te` drops HERE — off-GPU for the whole denoise (never co-resident with the DiT).
      // Stage 2: the DiT is resident ONLY for the denoise.
    {
        let dit = load_dit(state)?;
        use_dit(&dit, state)?;
        // Drain the denoise before `dit` frees and the VAE reuses the pool.
        sync()?;
    } // `dit` drops HERE — freed before the VAE is ever loaded.
      // Stage 3: the VAE decodes (the terminal component).
    let vae = load_vae(state)?;
    use_vae(&vae, state)
}

pub struct WanGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters applied to the DiT at first load (sc-10095) — folded (dense) or additive
    /// (packed q4/q8 tier).
    adapters: Vec<AdapterSpec>,
    /// Component-residency policy (epic 12732, sc-12757), resolved once at load via
    /// [`effective_offload_policy`] (honoring both `LoadSpec::offload_policy` and the family-wide
    /// `CANDLE_GEN_OFFLOAD=sequential` A/B override). [`OffloadPolicy::Resident`] keeps the cached
    /// [`Components`] warm; [`OffloadPolicy::Sequential`] drives the staged
    /// [`Pipeline::render_sequential`] (TE-offload + DiT-drop-before-VAE), bounding the denoise peak by
    /// keeping the ~11 GB bf16 UMT5 encoder (sc-12778) off-GPU. The resident [`components`](Self::components) cache
    /// stays untouched under `Sequential` — the staged path never populates it.
    offload: OffloadPolicy,
    components: Mutex<Option<Components>>,
}

impl WanGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `cached` recovers a poisoned lock (sc-9015) internally; `?` bridges the candle-side
        // `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for WanGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg("wan: prompt must not be empty".into()));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("wan: steps must be >= 1".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "wan: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // sc-12308: the 5B carries an area budget too (upstream `ti2v-5B` supports only `1280*704` /
        // `704*1280`), but this lane checked only the per-edge range and the multiple — so a
        // grid-aligned 1280×1280 validated and ran to an opaque OOM, exactly the sc-9028 hole the
        // 14B lane already closed. mlx's 5B did cap (it silently refit instead); rejecting here is
        // what makes the two backends agree on one geometry per request.
        let area = req.width as usize * req.height as usize;
        if area > MAX_AREA_5B {
            return Err(gen_core::Error::Msg(format!(
                "wan: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_5B} px \
                 (1280×704); reduce the resolution",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "wan: frames must satisfy frames % 4 == 1 (got {f})"
                )));
            }
            if f as usize > MAX_WAN_FRAMES {
                return Err(gen_core::Error::Msg(format!(
                    "wan: frames {f} exceeds the maximum {MAX_WAN_FRAMES}"
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load(&self.root, &self.device, self.adapters.clone());
        // Sequential offload (sc-12757): stage load→use→drop each heavy component so the denoise peak is
        // the DiT alone — the ~11 GB bf16 UMT5 encoder is off-GPU for the whole denoise and the DiT is
        // freed before the VAE loads. Resident (default): the cached `Components` bundle, unchanged path.
        // The staged path never populates the resident cache.
        let (frames, fps) = match self.offload {
            OffloadPolicy::Sequential => pipe.render_sequential(req, on_progress)?,
            OffloadPolicy::Resident => {
                let components = self.components(&pipe)?;
                pipe.render(req, &components, on_progress)?
            }
        };
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Wan2.2 TI2V-5B txt2video descriptor — the surface sc-3697 wires: CFG txt2video with a negative
/// prompt, UniPC / Euler samplers; no conditioning (image / VACE deferred). **LoRA/LoKr** apply at load
/// (sc-10095: folded on a dense tier, additive on a packed one). Advertises the Q4/Q8 packed tiers
/// (sc-10025) — pre-quantized snapshots the packed-detect loaders read directly (no on-the-fly quant).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![],
            // LoRA/LoKr apply at load (sc-10095): folded on a dense tier, or as additive residuals on a
            // packed q4/q8 tier (sc-10094). LoKr/LoHa on a packed tier is rejected at load (sc-10050/10051).
            supports_lora: true,
            supports_lokr: true,
            // Native flow samplers (curated `uni_pc` default / `euler`) + the epic 7114 P4 (sc-7124)
            // curated fold-in: the gen-core-only solvers over Wan's native flow σ schedule. The curated
            // `uni_pc` (sc-7296) is honored by Wan's OWN native UniPC; gen-core's VE-space `uni_pc`/
            // `dpmpp_2m` solvers are NOT routed through the fold-in (they would diverge from Wan's
            // diffusers FLOW-SNR parity). Legacy `unipc` retained as an alias for recipe back-compat. No
            // scheduler axis (the flow shift is the `scheduler_shift` knob).
            samplers: vec![
                "uni_pc",
                "euler",
                "euler_ancestral",
                "heun",
                "dpmpp_sde",
                "ddim",
                "unipc",
            ],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            // Per-side floor 480 (= a 15×15 latent-token grid): below it the z48 vae22's coarse
            // effective 32× stride starves the DiT, which renders rainbow garbage at ANY flow-shift
            // (dense + packed alike, sc-10306). Enforced by `Capabilities::validate_request`.
            min_size: MIN_SIZE,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // The TI2V-5B honors `OffloadPolicy::Sequential` (epic 12732, sc-12757): the staged
            // `render_sequential` keeps the ~11 GB bf16 UMT5 encoder off-GPU for the whole denoise and
            // frees the dense DiT before the VAE loads, bounding the pre-decode peak. Advertised so the
            // worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            supports_streaming: false,
            supports_multi_speaker: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
        },
    }
}

/// Construct a lazy candle Wan generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at
/// a `Wan-AI/Wan2.2-TI2V-5B-Diffusers` dense snapshot OR a pre-quantized MLX tier
/// (`SceneWorks/wan2.2-ti2v-5b-mlx` q4/q8) — the packed-detect loaders (sc-10025) read whichever the
/// dir holds. `spec.quantize` is a no-op: the tier is already packed (or dense), never requantized at
/// load. **LoRA/LoKr adapters** apply at first `generate` (sc-10095: folded on a dense tier, additive on
/// a packed one); control / VACE / IP-adapter overlays are still rejected (not wired).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(build_generator(spec)?))
}

/// The concrete constructor behind [`load`] — validates the spec surface and resolves the residency
/// policy, returning the concrete [`WanGenerator`] so the offload-policy wiring is unit-testable without
/// a `dyn Generator` downcast (sc-12757, mirroring the A14B's `build_generator`).
fn build_generator(spec: &LoadSpec) -> gen_core::Result<WanGenerator> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "wan expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
                    .into(),
            ));
        }
    };
    // Adapters are applied at first load (sc-10095): the packed-vs-dense branch lives in
    // `Pipeline::build_dit`. No `spec.quantize` reject (sc-10025): the quant matrix is packed-tier, not
    // on-the-fly — a q4/q8 tier is pre-quantized (the packed-detect loaders read its `.scales`), a dense
    // tier loads dense, so `spec.quantize` is a no-op tier-select marker resolved worker-side (ltx sc-9417).
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle wan does not support image / VACE conditioning yet (txt2video only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    // Resolve the residency policy once (sc-12757): honors both `spec.offload_policy` and the
    // family-wide `CANDLE_GEN_OFFLOAD=sequential` A/B override.
    let offload = effective_offload_policy(spec.offload_policy);
    Ok(WanGenerator {
        descriptor: descriptor(),
        root,
        device,
        adapters: spec.adapters.clone(),
        offload,
        components: Mutex::new(None),
    })
}

candle_gen::register_generators! {
    pub(crate) const TI2V_REGISTRATION = descriptor => load
}

/// Add all Candle Wan generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(TI2V_REGISTRATION)
        .register_generator(wan14b::T2V_14B_REGISTRATION)
        .register_generator(wan14b::I2V_14B_REGISTRATION)
        .register_generator(model_vace::VACE_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit Candle Wan provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            explicit_generators,
            [
                "wan2_2_ti2v_5b",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan_vace",
            ]
        );
        assert_eq!(explicit_trainers, ["wan2_2_t2v_14b"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors as cst;
    use candle_gen::gen_core::ConditioningKind;
    use std::collections::HashMap;

    #[test]
    fn combined_conditioning_latents_is_checked() {
        assert_eq!(super::combined_conditioning_latents(1025, 0), Some(257));
        assert_eq!(super::combined_conditioning_latents(5, 255), Some(257));
        assert_eq!(super::combined_conditioning_latents(5, usize::MAX), None);
        assert_eq!(super::combined_conditioning_latents(0, 0), None);
    }

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("wan is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "wan");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(d.capabilities.samplers.contains(&"unipc")); // legacy alias retained
        assert!(d.capabilities.samplers.contains(&"euler"));
    }

    /// sc-12308: the 5B's own area budget is enforced here. `validate` previously checked only the
    /// per-edge range and the ÷32 multiple, so a grid-aligned far-over-envelope request reached the
    /// DiT and died on an opaque OOM — while mlx's 5B capped (by silently refitting). The 5B keeps
    /// the 901 120 budget upstream gives `ti2v-5B`; it must NOT inherit the 14B family's larger one.
    #[test]
    fn validate_enforces_max_area_5b() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            frames: Some(17),
            ..Default::default()
        };

        // The 5B's grid is 32, so its 720p IS `1280×704` — exactly at the cap, and accepted.
        assert_eq!(1280 * 704, MAX_AREA_5B);
        assert!(g
            .validate(&GenerationRequest {
                width: 1280,
                height: 704,
                ..base.clone()
            })
            .is_ok());

        // Over the cap with both edges grid-aligned and within the per-edge range: rejected by the
        // area check specifically, with a message naming the cap.
        let err = g
            .validate(&GenerationRequest {
                width: 1280,
                height: 1280,
                ..base.clone()
            })
            .expect_err("over-area request must be rejected");
        assert!(
            err.to_string().contains("max area"),
            "actionable message: {err}"
        );

        // The 5B must not drift onto the 14B family's budget: `1280×720` is 921 600 px, over the
        // 5B's cap AND off its 32-px grid, so it stays rejected on this lane.
        const { assert!(config::MAX_AREA_14B > MAX_AREA_5B) };
        assert!(g
            .validate(&GenerationRequest {
                width: 1280,
                height: 720,
                ..base
            })
            .is_err());
    }

    #[test]
    fn validate_accepts_txt2video_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 512,
            height: 512,
            count: 1,
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        assert!(g
            .validate(&GenerationRequest {
                frames: Some(1025),
                ..ok.clone()
            })
            .is_ok());
        let over = g
            .validate(&GenerationRequest {
                frames: Some(1029),
                ..ok.clone()
            })
            .expect_err("1029 must exceed the Wan frame ceiling");
        assert!(over.to_string().contains("maximum 1025"), "{over}");
        // Legacy `unipc` spelling stays accepted (sc-7296 alias).
        assert!(g
            .validate(&GenerationRequest {
                sampler: Some("unipc".into()),
                ..ok.clone()
            })
            .is_ok());
        // Each bad case spreads from the valid `ok` so it is rejected for its OWN reason, not an
        // unrelated default.
        for bad in [
            // empty prompt
            GenerationRequest {
                prompt: String::new(),
                ..ok.clone()
            },
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                frames: Some(16),
                ..ok.clone()
            },
            // size not a multiple of 32 (500 is in-range but 500 % 32 != 0)
            GenerationRequest {
                width: 500,
                ..ok.clone()
            },
            // below the per-side min-size floor (sc-10306): 320² is 32-aligned but under 480 → the z48
            // token grid is too coarse to converge, so the descriptor rejects it up front.
            GenerationRequest {
                width: 320,
                height: 320,
                ..ok.clone()
            },
            // zero steps
            GenerationRequest {
                steps: Some(0),
                ..ok.clone()
            },
            // unadvertised sampler
            GenerationRequest {
                sampler: Some("dpmpp2m".into()),
                ..ok.clone()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let req = GenerationRequest {
            prompt: " \t\n ".into(),
            frames: Some(17),
            ..Default::default()
        };
        assert!(g.validate(&req).is_err());
    }

    #[test]
    fn load_accepts_lora_and_quant() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // LoRA/LoKr is wired (sc-10095) — load is lazy, so attaching adapters resolves OK (the fold /
        // additive install happens at the first `generate`).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA is accepted (applied lazily)");
        // Quant is a no-op tier-select marker (packed-detect load, sc-10025), not a reject — a q4/q8
        // tier is pre-quantized, so `spec.quantize` no longer errors (lazy load, no fs touch here).
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "quant is accepted (packed-tier select, no on-the-fly quant)"
        );
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    // ---- packed-tier adapter routing (sc-10095) -------------------------------------------------

    use candle_gen::candle_nn::VarMap;

    /// A tiny Wan DiT config — the dit_train shape (z16, 2 layers), enough to exercise the packed-detect
    /// + additive-route path in `Pipeline::build_dit` cheaply on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 2,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    /// Build a tiny **packed** transformer tier on disk under `{root}/transformer/`: a randomized dense
    /// DiT map, MLX-affine-packed by the sc-10026 producer, written as `model.safetensors` (+ a
    /// `quantize_config.json`) — the exact packed-detect layout the sc-10025 seam loads.
    fn write_packed_transformer(root: &Path, cfg: &TransformerConfig) {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let _ = WanTransformer::new(cfg, vb).unwrap();
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), &dev).unwrap())
                .unwrap();
        }
        let map: HashMap<String, Tensor> = {
            let data = vm.data().lock().unwrap();
            data.iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
                .collect()
        };
        let (packed, _n) = candle_tier_build::pack_transformer_component(map, 4).unwrap();
        let dir = root.join("transformer");
        std::fs::create_dir_all(&dir).unwrap();
        cst::save(&packed, dir.join("model.safetensors")).unwrap();
        std::fs::write(dir.join("quantize_config.json"), "{\"bits\":4}").unwrap();
    }

    fn tiny_pipeline(root: &Path, adapters: Vec<AdapterSpec>) -> Pipeline {
        Pipeline {
            adapters,
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: tiny_cfg(),
            vae_cfg: VaeConfig::ti2v_5b(),
            root: root.to_path_buf(),
            device: Device::Cpu,
        }
    }

    /// `build_dit` loads a packed tier through the packed path (`is_packed()`), and with a LoRA it
    /// installs the residual additively (the base stays packed — no dense weight materialized) rather
    /// than folding, which a packed tier can't support. The core sc-10095 routing on a real tier layout.
    #[test]
    fn build_dit_routes_packed_tier_through_additive() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let root = std::env::temp_dir().join(format!("sc10095_5b_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        write_packed_transformer(&root, &cfg);

        // No adapters: the packed tier loads packed, unadapted.
        let base = tiny_pipeline(&root, vec![]).build_dit().unwrap();
        assert!(
            base.is_packed(),
            "packed tier must load through the packed path"
        );

        // A LoRA on `blocks.0.attn1.to_q`: applies additively, base stays packed.
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "blocks.0.attn1.to_q.lora_A.weight".into(),
            (Tensor::randn(0f32, 1f32, (4, cfg.dim), &dev).unwrap() * 0.1).unwrap(),
        );
        m.insert(
            "blocks.0.attn1.to_q.lora_B.weight".into(),
            (Tensor::randn(0f32, 1f32, (cfg.dim, 4), &dev).unwrap() * 0.1).unwrap(),
        );
        let lora_path = root.join("lora.safetensors");
        cst::save(&m, &lora_path).unwrap();
        let specs = vec![candle_gen::gen_core::AdapterSpec::new(
            lora_path,
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        )];
        let adapted = tiny_pipeline(&root, specs).build_dit().unwrap();
        assert!(
            adapted.is_packed(),
            "the additive LoRA must not un-pack the base"
        );
        // (The numeric forward shift is a CUDA-only check — the DiT runs bf16, and CPU has no bf16
        // matmul; that's the on-device sc-10026 gate. The QLinear-level additive-on-packed forward is
        // covered on CPU in `quant::tests::additive_lora_on_packed_shifts_and_finite`.)

        // A LoRA that matches NO projection is surfaced by the packed zero-match guard (proving the
        // additive install actually ran, not a silent no-op) — a misconfigured file hard-errors rather
        // than rendering unadapted.
        let mut bogus: HashMap<String, Tensor> = HashMap::new();
        bogus.insert(
            "blocks.99.attn1.to_q.lora_A.weight".into(),
            Tensor::randn(0f32, 1f32, (4, cfg.dim), &dev).unwrap(),
        );
        bogus.insert(
            "blocks.99.attn1.to_q.lora_B.weight".into(),
            Tensor::randn(0f32, 1f32, (cfg.dim, 4), &dev).unwrap(),
        );
        let bogus_path = root.join("bogus.safetensors");
        cst::save(&bogus, &bogus_path).unwrap();
        let bogus_specs = vec![candle_gen::gen_core::AdapterSpec::new(
            bogus_path,
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        )];
        assert!(
            tiny_pipeline(&root, bogus_specs).build_dit().is_err(),
            "a LoRA matching no packed projection must hard-error (zero-match guard)"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    // ── sc-12757: dense sequential component offload (TE/VAE off-GPU) — Pillar 1 ──

    /// The 5B must now advertise `supports_sequential_offload` so the worker's fit-gate can tell "the
    /// staged path bounds peak VRAM here" from a no-op fallback (sc-11126 contract). It was `false`
    /// before this story — the TI2V-5B was NOT touched by the A14B's sc-12733.
    #[test]
    fn descriptor_advertises_sequential_offload() {
        assert!(
            descriptor().capabilities.supports_sequential_offload,
            "the TI2V-5B must advertise sequential offload (sc-12757)"
        );
    }

    /// The load path resolves the residency policy from `LoadSpec::offload_policy` via
    /// [`effective_offload_policy`]: the default spec stays `Resident` (cached-components, unchanged
    /// path), an explicit `Sequential` spec flips the generator onto the staged `render_sequential`.
    #[test]
    fn load_resolves_offload_policy_from_spec() {
        let resident = build_generator(&LoadSpec::new(WeightsSource::Dir("/snap".into()))).unwrap();
        assert_eq!(resident.offload, OffloadPolicy::Resident);

        let sequential = build_generator(
            &LoadSpec::new(WeightsSource::Dir("/snap".into()))
                .with_offload_policy(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_eq!(sequential.offload, OffloadPolicy::Sequential);
    }

    /// A liveness witness for the sequential-offload residency tests, mirroring the drop-order witnesses
    /// in `wan14b`'s tests: it bumps a shared live-counter on construction and drops it on `Drop`,
    /// recording the peak concurrency and an ordered load/use/drop log.
    struct LiveTracker {
        live: std::cell::Cell<usize>,
        peak: std::cell::Cell<usize>,
        log: std::cell::RefCell<Vec<&'static str>>,
    }

    impl LiveTracker {
        fn new() -> Self {
            Self {
                live: std::cell::Cell::new(0),
                peak: std::cell::Cell::new(0),
                log: std::cell::RefCell::new(Vec::new()),
            }
        }
        fn born(&self, tag: &'static str) {
            self.live.set(self.live.get() + 1);
            if self.live.get() > self.peak.get() {
                self.peak.set(self.live.get());
            }
            self.log.borrow_mut().push(tag);
        }
        fn died(&self, tag: &'static str) {
            self.live.set(self.live.get() - 1);
            self.log.borrow_mut().push(tag);
        }
        fn note(&self, tag: &'static str) {
            self.log.borrow_mut().push(tag);
        }
    }

    /// Stands in for a loaded heavy component (UMT5 / DiT / VAE): its lifetime on the live-counter is
    /// exactly that component's GPU-residency window in `staged_sequential`.
    struct CompWitness<'a> {
        tracker: &'a LiveTracker,
        drop_tag: &'static str,
    }

    impl<'a> CompWitness<'a> {
        fn new(tracker: &'a LiveTracker, born_tag: &'static str, drop_tag: &'static str) -> Self {
            tracker.born(born_tag);
            Self { tracker, drop_tag }
        }
    }

    impl Drop for CompWitness<'_> {
        fn drop(&mut self) {
            self.tracker.died(self.drop_tag);
        }
    }

    /// The Pillar-1 invariant (sc-12757): at most ONE heavy component is resident at a time, and the
    /// stages drop in order — the TE is off-GPU before the denoise runs, and the DiT drops before the
    /// VAE loads. Driven through the production `staged_sequential` with drop-order witnesses (candle's
    /// cudarc pool makes `nvidia-smi` blind to a drop, so residency is asserted structurally, not by a
    /// VRAM read). `use_vae` returns a sentinel to prove the value threads back out.
    #[test]
    fn sequential_stages_are_never_co_resident_and_drop_in_order() {
        let tracker = LiveTracker::new();
        let mut st = ();
        let out = staged_sequential(
            &mut st,
            |_st| Ok(CompWitness::new(&tracker, "load-te", "drop-te")),
            |_w, _st| {
                tracker.note("use-te");
                Ok(())
            },
            |_st| Ok(CompWitness::new(&tracker, "load-dit", "drop-dit")),
            |_w, _st| {
                tracker.note("use-dit");
                Ok(())
            },
            |_st| Ok(CompWitness::new(&tracker, "load-vae", "drop-vae")),
            |_w, _st| {
                tracker.note("use-vae");
                Ok(42usize)
            },
            || Ok(()),
        );
        assert_eq!(out.unwrap(), 42, "the decode result must thread back out");
        assert_eq!(
            tracker.peak.get(),
            1,
            "at most ONE heavy component may be GPU-resident at a time (the whole Pillar-1 win)"
        );
        assert_eq!(
            *tracker.log.borrow(),
            vec![
                "load-te", "use-te", "drop-te", //
                "load-dit", "use-dit", "drop-dit", //
                "load-vae", "use-vae", "drop-vae",
            ],
            "the stages must load → use → drop strictly in order"
        );
        // Explicit drop-order sub-assertions (the story's two named liveness guarantees).
        let log = tracker.log.borrow();
        let at = |tag: &str| log.iter().position(|x| *x == tag).unwrap();
        assert!(
            at("drop-te") < at("use-dit"),
            "the ~11 GB bf16 UMT5 encoder must be off-GPU BEFORE the denoise (use-dit) runs"
        );
        assert!(
            at("drop-dit") < at("load-vae"),
            "the DiT must be dropped BEFORE the VAE loads"
        );
    }

    /// Mutation-check (sc-12757 acceptance): force the TE resident across the DiT load+denoise and
    /// confirm the never-co-resident assertion regresses — proving the passing test above is not a
    /// default-value false green. This is the exact dead-weight-resident bug the story removes (the old
    /// `Components` held the f32 UMT5 for the whole render): binding `te` and `dit` in one scope
    /// co-resides them, and the SAME liveness witness the passing test relies on now reports peak
    /// concurrency 2, so its `peak == 1` assertion goes RED.
    #[test]
    fn forcing_te_resident_regresses_the_never_co_resident_assertion() {
        let tracker = LiveTracker::new();
        {
            // MUTATION: the TE is NOT dropped before the DiT loads / denoise runs.
            let _te = CompWitness::new(&tracker, "load-te", "drop-te");
            let _dit = CompWitness::new(&tracker, "load-dit", "drop-dit");
            tracker.note("te-resident-during-denoise");
        }
        assert_eq!(
            tracker.peak.get(),
            2,
            "the forced-resident mutation co-resides the TE + DiT"
        );
        assert!(
            tracker.peak.get() > 1,
            "the never-co-resident assertion (peak == 1) MUST fail under the TE-resident mutation — \
             the passing test genuinely discriminates residency, it is not a false green"
        );
    }

    /// The sc-12195 eviction sync, applied per stage boundary (sc-12757): the boundary `sync` runs after
    /// each heavy component is used and **before** it drops (and before the next loads) — draining
    /// in-flight kernels so the freed allocator pool is never reused under them. The terminal VAE decode
    /// has no trailing sync (nothing loads after it). Mirrors `wan14b`'s boundary-sync ordering witness.
    #[test]
    fn sequential_syncs_before_each_component_drops() {
        let tracker = LiveTracker::new();
        let mut st = ();
        staged_sequential(
            &mut st,
            |_st| Ok(CompWitness::new(&tracker, "load-te", "drop-te")),
            |_w, _st| {
                tracker.note("use-te");
                Ok(())
            },
            |_st| Ok(CompWitness::new(&tracker, "load-dit", "drop-dit")),
            |_w, _st| {
                tracker.note("use-dit");
                Ok(())
            },
            |_st| Ok(CompWitness::new(&tracker, "load-vae", "drop-vae")),
            |_w, _st| {
                tracker.note("use-vae");
                Ok(())
            },
            || {
                tracker.note("sync");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            *tracker.log.borrow(),
            vec![
                "load-te", "use-te", "sync", "drop-te", //
                "load-dit", "use-dit", "sync", "drop-dit", //
                "load-vae", "use-vae", "drop-vae",
            ],
            "each heavy component must be synced (kernels drained) before it drops and the next loads"
        );
    }

    /// A load failure on the DiT still drops the (already-used-and-synced) TE via scope drop on the `?`
    /// path — no leak, and the error propagates. The VAE is never loaded.
    #[test]
    fn sequential_propagates_a_dit_load_failure_after_dropping_te() {
        let tracker = LiveTracker::new();
        let mut st = ();
        let out: CResult<()> = staged_sequential(
            &mut st,
            |_st| Ok(CompWitness::new(&tracker, "load-te", "drop-te")),
            |_w, _st| Ok(()),
            |_st| Err(CandleError::Msg("DiT OOM".into())),
            |_w: &CompWitness, _st| Ok(()),
            |_st| Ok(CompWitness::new(&tracker, "load-vae", "drop-vae")),
            |_w, _st| Ok(()),
            || Ok(()),
        );
        assert!(matches!(out, Err(CandleError::Msg(_))));
        assert_eq!(
            *tracker.log.borrow(),
            vec!["load-te", "drop-te"],
            "the TE must have dropped before the DiT load was even attempted; the VAE never loads"
        );
        assert_eq!(tracker.peak.get(), 1);
    }
}
