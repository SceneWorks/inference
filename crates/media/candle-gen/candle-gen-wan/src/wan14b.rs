//! The Wan2.2 **A14B dual-expert MoE** video providers (sc-5174) — the candle (Windows/CUDA) siblings
//! of `mlx-gen-wan`'s `wan2_2_t2v_14b` / `wan2_2_i2v_14b`. Both register as `backend = "candle"`,
//! [`Modality::Video`].
//!
//! Wan2.2's "MoE" is **two complete `WanTransformer3DModel` checkpoints**, not token routing: a
//! **high-noise** expert (`transformer/`) and a **low-noise** expert (`transformer_2/`). A single
//! flow-match scheduler drives the denoise; each step picks the high expert while the integer timestep
//! is `≥ boundary·1000` (T2V `0.875`, I2V `0.900`) and the low expert below it, switching the
//! transformer, its (per-expert) text context, and its guidance scale together (T2V 3.0/4.0, I2V
//! 3.5/3.5). The experts share the dimension-parametric [`WanTransformer`] (loaded with
//! [`TransformerConfig::t2v_14b`]/[`i2v_14b`](TransformerConfig::i2v_14b)) and the [`crate::vae16`] z16
//! VAE — *not* the 5B's z48 VAE (the 14B emits 16-channel latents).
//!
//! **T2V** (`wan2_2_t2v_14b`): pure text→video. **I2V** (`wan2_2_i2v_14b`): channel-concat conditioning
//! — the reference image's first-frame z16 latent + a temporal mask form a 20-channel `y` appended to
//! the 16-channel noise latent (in_dim 36) every forward (the image enters via the channels, not noise).
//!
//! **Dtypes:** the experts, UMT5 (sc-12778), and the z16 VAE (sc-12818) all run **bf16** — the experts
//! and VAE with their norms/modulation (and the VAE's channel-L2 + attention softmax) upcast to f32,
//! mirroring the 5B. The VAE decode **streams one latent frame at a time** (sc-5176) to bound the
//! decode-stage peak — the heavier-than-5B fix the story (sc-5174) requires — and the bf16 VAE halves
//! that stage's otherwise-fixed ~30 GiB floor so the 1280×720/81f A14B decode fits a 24 GiB card.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{safetensors as cst, DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::{CancelFlag, LoadPhase};
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, MoeExpert,
    OffloadPolicy, Progress, Quant, WeightsSource,
};
use candle_gen::{check_cancel, effective_offload_policy, CandleError, Result as CResult};

use crate::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, DEFAULT_FPS_14B, DEFAULT_FRAMES_14B,
    DEFAULT_STEPS_14B, I2V_14B_BOUNDARY, I2V_14B_FLOW_SHIFT, I2V_14B_GUIDANCE_HIGH,
    I2V_14B_GUIDANCE_LOW, MAX_AREA_14B, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B, NEGATIVE_FALLBACK,
    NUM_TRAIN_TIMESTEPS, SIZE_MULTIPLE_14B, T2V_14B_BOUNDARY, T2V_14B_FLOW_SHIFT,
    T2V_14B_GUIDANCE_HIGH, T2V_14B_GUIDANCE_LOW, VAE16_STRIDE_SPATIAL, VAE16_STRIDE_TEMPORAL,
};
use crate::pipeline::{cfg, create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::{FlowScheduler, Sampler};
use crate::text_encoder::Umt5Encoder;
use crate::transformer::WanTransformer;
use crate::vae16::WanVae16;

/// The experts run bf16 (the diffusers fp32 weights load as bf16, the 5B regime); UMT5 runs bf16
/// (sc-12778 — halving the f32 encoder resident + its ENCODE-stage transient; the DiT `embed_text`
/// already casts the context to bf16, so this removes the old f32→bf16 boundary. The A14B is
/// decode-bound so the peak win is smaller than the 5B's, but the resident encoder still halves).
///
/// The z16 VAE runs **bf16** (sc-12818): rigorously measured (driver `CU_MEMPOOL_ATTR_USED_MEM_HIGH`,
/// the accurate concurrent-live peak), the sequential A14B decode's TRUE peak was a **fixed ~30.1 GiB
/// floor, independent of the VAE spatial-tile budget** (weights + the un-tileable f32 decode
/// activations, not something tiling can shrink). Running the z16 VAE bf16 ~halves that floor → it fits
/// a 24 GiB card. bf16 keeps f32's 8-bit exponent (no fp16 overflow risk), and the L2/softmax
/// reductions stay f32-reduced in [`WanVae16`], so decode parity holds (GPU-checked ≥35 dB PSNR vs f32).
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::BF16;
const Z_DIM: usize = 16;

/// Which A14B model this generator serves — selects in_dim (16 vs 36), the MoE knobs, and whether the
/// VAE carries an encoder (I2V conditioning).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    T2v,
    I2v,
}

impl Variant {
    fn id(self) -> &'static str {
        match self {
            Variant::T2v => MODEL_ID_T2V_14B,
            Variant::I2v => MODEL_ID_I2V_14B,
        }
    }

    fn dit_cfg(self) -> TransformerConfig {
        match self {
            Variant::T2v => TransformerConfig::t2v_14b(),
            Variant::I2v => TransformerConfig::i2v_14b(),
        }
    }

    /// `(boundary, default flow-shift, guidance_low, guidance_high)`.
    fn moe_knobs(self) -> (f64, f64, f32, f32) {
        match self {
            Variant::T2v => (
                T2V_14B_BOUNDARY,
                T2V_14B_FLOW_SHIFT,
                T2V_14B_GUIDANCE_LOW,
                T2V_14B_GUIDANCE_HIGH,
            ),
            Variant::I2v => (
                I2V_14B_BOUNDARY,
                I2V_14B_FLOW_SHIFT,
                I2V_14B_GUIDANCE_LOW,
                I2V_14B_GUIDANCE_HIGH,
            ),
        }
    }
}

/// True when classifier-free guidance is actually active: the negative/uncond branch only changes the
/// output at `guidance > 1.0`. At `guidance <= 1.0` the CFG combine `neg + g·(pos − neg)` reduces to
/// `pos` (exactly `pos` at 1.0), so the negative UMT5 encode + per-expert projection + per-step forward
/// are pure waste and are skipped (sc-8993). Kept as one predicate so the encode-time gate and the
/// per-step gate can never diverge.
fn cfg_active(guidance: f64) -> bool {
    guidance > 1.0
}

#[derive(Clone)]
struct Components {
    te: Arc<Umt5Encoder>,
    /// `transformer/` — the **high-noise** expert (timestep ≥ boundary).
    high: Arc<WanTransformer>,
    /// `transformer_2/` — the **low-noise** expert (timestep < boundary).
    low: Arc<WanTransformer>,
    vae: Arc<WanVae16>,
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across encodes (sc-8991 /
    /// F-011) instead of re-parsing `tokenizer.json` per prompt/branch.
    tok: Arc<candle_gen::gen_core::tokenizer::TextTokenizer>,
}

struct Pipeline {
    te_cfg: TextEncoderConfig,
    dit_cfg: TransformerConfig,
    vae_cfg: Vae16Config,
    variant: Variant,
    root: PathBuf,
    device: Device,
    /// Trained LoRA/LoKr adapters to merge into the experts at load (sc-5167). Each is routed to the
    /// high and/or low expert by its [`AdapterSpec::moe_expert`].
    adapters: Vec<AdapterSpec>,
    /// In-place ComfyUI expert files (epic 10451 Phase 2c, sc-10671). When set, the two experts are
    /// built from these files (scaled-fp8 dequant + key remap, see [`crate::comfyui`]) instead of the
    /// snapshot's `transformer/` + `transformer_2/`. The UMT5 TE + VAE are read in place too when the
    /// spec carries their files (sc-10909), else from `root`; the tiny tokenizer always comes from
    /// `root`. `None` on the registry path.
    comfyui: Option<std::sync::Arc<crate::comfyui::ComfyuiExperts>>,
}

impl Pipeline {
    fn load(root: &Path, device: &Device, variant: Variant, adapters: Vec<AdapterSpec>) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: variant.dit_cfg(),
            vae_cfg: Vae16Config::wan21(),
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            adapters,
            comfyui: None,
        }
    }

    /// Same as [`load`](Self::load) but with the two experts sourced from an in-place ComfyUI install
    /// (sc-10671). `root` is the resident Wan snapshot tier supplying the UMT5 TE / VAE / tokenizer;
    /// `comfyui` names the user's two ComfyUI expert files, read in place.
    fn load_comfyui(
        root: &Path,
        device: &Device,
        variant: Variant,
        comfyui: std::sync::Arc<crate::comfyui::ComfyuiExperts>,
    ) -> Self {
        Self {
            te_cfg: TextEncoderConfig::umt5_xxl(),
            dit_cfg: variant.dit_cfg(),
            vae_cfg: Vae16Config::wan21(),
            variant,
            root: root.to_path_buf(),
            device: device.clone(),
            adapters: Vec::new(),
            comfyui: Some(comfyui),
        }
    }

    /// Build one expert from an in-place ComfyUI file (sc-10671): load its native tensor map on CPU,
    /// remap keys + dequant the scaled-fp8 weights ([`crate::comfyui`]) into the diffusers schema, then
    /// build via `VarBuilder::from_tensors` (the in-memory path the packed/adapter loads already use).
    fn build_expert_comfyui(&self, file: &Path) -> CResult<WanTransformer> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::remap_and_dequant_comfyui_expert(map, DIT_DTYPE)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok(WanTransformer::new(&self.dit_cfg, vb)?)
    }

    /// Build the UMT5 text encoder from an in-place ComfyUI file (sc-10909): load its native tensor map
    /// on CPU, dequant the companion scaled-fp8 weights ([`crate::comfyui::dequant_comfyui_umt5`], no
    /// key remap — the ComfyUI file already carries the HF `UMT5EncoderModel` keys), then build via
    /// `VarBuilder::from_tensors` at [`ENC_DTYPE`] (bf16, sc-12778 — matching the snapshot TE; the
    /// scaled-fp8 dequant computes in f32 then casts to the bf16 target, as the DiT experts already do).
    fn build_te_comfyui(&self, file: &Path) -> CResult<Umt5Encoder> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::dequant_comfyui_umt5(map, ENC_DTYPE)?;
        let vb = VarBuilder::from_tensors(map, ENC_DTYPE, &self.device);
        Ok(Umt5Encoder::new(&self.te_cfg, vb)?)
    }

    /// Build the z16 VAE from an in-place ComfyUI file (sc-10909): load its native tensor map on CPU,
    /// remap the native WAN-VAE keys to the diffusers schema
    /// ([`crate::comfyui::remap_vae_wan_to_diffusers`], values pass through as bf16), then build via
    /// `VarBuilder::from_tensors` at [`VAE_DTYPE`] (**bf16**, sc-12818 — the weights load bf16, the
    /// decode-floor win; `get` casts each tensor to that dtype). I2V builds the encoder too (the
    /// conditioning image's first-frame latent).
    fn build_vae_comfyui(&self, file: &Path) -> CResult<WanVae16> {
        let map = cst::load(file, &Device::Cpu)?;
        let map = crate::comfyui::remap_vae_wan_to_diffusers(map)?;
        let vb = VarBuilder::from_tensors(map, VAE_DTYPE, &self.device);
        match self.variant {
            Variant::I2v => Ok(WanVae16::new_with_encoder(&self.vae_cfg, vb)?),
            Variant::T2v => Ok(WanVae16::new(&self.vae_cfg, vb)?),
        }
    }

    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        // Shared Wan component loader (sc-9000 / F-020); the crafted snapshot description (which names
        // the expected A14B variant) stays local.
        let desc = match self.variant {
            Variant::T2v => "Wan2.2-T2V-A14B diffusers",
            Variant::I2v => "Wan2.2-I2V-A14B diffusers",
        };
        crate::text_encode::component_vb(&self.root, sub, dtype, &self.device, "wan-14b", desc)
    }

    /// Build one expert from its `sub` dir, applying any adapter whose [`AdapterSpec::moe_expert`]
    /// targets it (`Some(expert)` or `None` = shared). The adapter application differs by tier:
    ///
    /// - **Dense** tier: the delta is FOLDED into the dense weights before build
    ///   ([`crate::adapters::merge_adapters`], f32 math, `VarBuilder::from_tensors`) — the
    ///   merge-not-residual pattern the SDXL/Z-Image ports established, byte-identical to before.
    ///   `merge_adapters` hard-errors on its own zero-match (the report is otherwise discarded, F-051).
    /// - **Packed** q4/q8 tier (sc-10095): a packed tier has **no dense `W`** to fold into, so the
    ///   adapters attach as forward-time **additive** residuals on the packed `QLinear`
    ///   ([`crate::adapters::install_additive`], sc-10094) — the base weight stays q4/q8. LoKr/LoHa on a
    ///   packed tier is rejected there (deferred to sc-10050/10051).
    ///
    /// Returns the expert plus `Some(applied)` on the packed path (the count of attached residuals, for
    /// the caller's cross-expert zero-match guard) or `None` on the dense/no-adapter paths (which
    /// self-guard). With no adapter for this expert the fast mmap build is used unchanged.
    fn build_expert(
        &self,
        sub: &str,
        expert: MoeExpert,
    ) -> CResult<(WanTransformer, Option<usize>)> {
        let specs: Vec<AdapterSpec> = self
            .adapters
            .iter()
            .filter(|s| s.moe_expert.is_none_or(|e| e == expert))
            .cloned()
            .collect();
        let vb = self.component_vb(sub, DIT_DTYPE)?;
        // Packed-tier marker: the sc-10025 seam packs every DiT Linear (incl. `proj_out`), so a
        // `proj_out.scales` sibling is present iff this is a pre-quantized q4/q8 tier.
        let packed = vb.contains_tensor("proj_out.scales");
        if packed {
            let mut dit = WanTransformer::new(&self.dit_cfg, vb)?;
            if specs.is_empty() {
                return Ok((dit, Some(0)));
            }
            // Additive install on the packed base — no dense weight materialized (sc-10094/10095).
            let report = crate::adapters::install_additive(&mut dit, &specs, expert)?;
            return Ok((dit, Some(report.applied)));
        }
        if specs.is_empty() {
            return Ok((WanTransformer::new(&self.dit_cfg, vb)?, None));
        }
        // Dense tier + adapters: fold the delta into the dense weights before building (the legacy
        // merge-not-residual fast path). `merge_adapters` hard-errors on its own zero-match.
        drop(vb);
        let mut map = crate::text_encode::load_component_map(&self.root, sub, "wan-14b")?;
        crate::adapters::merge_adapters(&mut map, &specs)?;
        let vb = VarBuilder::from_tensors(map, DIT_DTYPE, &self.device);
        Ok((WanTransformer::new(&self.dit_cfg, vb)?, None))
    }

    /// Build the UMT5 text encoder (bf16, sc-12778) — the ~11 GB phase-A component (was ~21 GB f32).
    /// In-place ComfyUI mode (sc-10909) reads it from the user's tree (scaled-fp8 dequant); else the
    /// snapshot `text_encoder/`.
    /// A single home shared by the resident [`load_components`](Self::load_components) build and the
    /// sequential-offload [`render_sequential`](Self::render_sequential) stage (sc-12733).
    fn load_te(&self) -> CResult<Umt5Encoder> {
        match self.comfyui.as_ref().and_then(|c| c.te_file.as_deref()) {
            Some(te_file) => self.build_te_comfyui(te_file),
            None => Ok(Umt5Encoder::new(
                &self.te_cfg,
                self.component_vb("text_encoder", ENC_DTYPE)?,
            )?),
        }
    }

    /// Build one 14B expert (~8-9 GB bf16) from either the in-place ComfyUI file (sc-10671) or the
    /// snapshot subdir (`transformer/` = high, `transformer_2/` = low). Returns the transformer plus the
    /// packed-tier applied-residual count (`Some` iff packed, for the cross-expert zero-match guard;
    /// `None` on the dense/comfyui paths, which self-guard). Shared by the resident and staged paths so
    /// the two never diverge in how an expert is built (sc-12733).
    fn load_expert_staged(&self, expert: MoeExpert) -> CResult<(WanTransformer, Option<usize>)> {
        match &self.comfyui {
            Some(experts) => {
                let file = match expert {
                    MoeExpert::High => &experts.high_file,
                    MoeExpert::Low => &experts.low_file,
                };
                // The ComfyUI base lane folds no adapters, so `None` (zero-match guard inert).
                Ok((self.build_expert_comfyui(file)?, None))
            }
            None => {
                let sub = match expert {
                    MoeExpert::High => "transformer",
                    MoeExpert::Low => "transformer_2",
                };
                self.build_expert(sub, expert)
            }
        }
    }

    /// Build the z16 VAE (**bf16**, sc-12818 — the decode-floor win). In-place ComfyUI mode (sc-10909)
    /// reads it from the user's tree (native→diffusers key remap); else the snapshot `vae/`. I2V builds
    /// the encoder too (the conditioning image's first-frame latent). Shared by the resident and staged
    /// paths (sc-12733).
    fn load_vae(&self) -> CResult<WanVae16> {
        match self.comfyui.as_ref().and_then(|c| c.vae_file.as_deref()) {
            Some(vae_file) => self.build_vae_comfyui(vae_file),
            None => {
                let vae_vb = self.component_vb("vae", VAE_DTYPE)?;
                match self.variant {
                    // I2V needs the VAE encoder (the conditioning image's first-frame latent).
                    Variant::I2v => Ok(WanVae16::new_with_encoder(&self.vae_cfg, vae_vb)?),
                    Variant::T2v => Ok(WanVae16::new(&self.vae_cfg, vae_vb)?),
                }
            }
        }
    }

    /// Packed-tier zero-match guard (sc-10095): on the additive path a non-empty adapter set that
    /// attached NO residual across **either** packed expert is a format/prefix misconfiguration (the
    /// dense fold path self-guards inside `merge_adapters`, so it reports `None` and is exempt). Both
    /// experts share a tier, so each count is `Some` iff packed; the guard fires only when both experts
    /// were built (both `Some`) and neither matched — the staged path (which builds one expert at a time)
    /// runs it once both experts have loaded.
    fn adapter_zero_match_guard(
        &self,
        high_applied: Option<usize>,
        low_applied: Option<usize>,
    ) -> CResult<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        if let (Some(h), Some(l)) = (high_applied, low_applied) {
            if h + l == 0 {
                return Err(CandleError::Msg(format!(
                    "{}: {} LoRA adapter file(s) matched no projection on either packed expert — \
                     check the key format (expected PEFT `<path>.lora_A/B.weight` or kohya \
                     `lora_unet_<flat>` targeting the DiT attention/FFN Linears)",
                    self.variant.id(),
                    self.adapters.len()
                )));
            }
        }
        Ok(())
    }

    fn load_components(&self) -> CResult<Components> {
        let te = self.load_te()?;
        // transformer/ = high-noise expert, transformer_2/ = low-noise expert (diffusers WanPipeline).
        let (high, high_applied) = self.load_expert_staged(MoeExpert::High)?;
        let (low, low_applied) = self.load_expert_staged(MoeExpert::Low)?;
        self.adapter_zero_match_guard(high_applied, low_applied)?;
        let vae = self.load_vae()?;
        let tok = crate::text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan-14b")?;
        Ok(Components {
            te: Arc::new(te),
            high: Arc::new(high),
            low: Arc::new(low),
            vae: Arc::new(vae),
            tok: Arc::new(tok),
        })
    }

    /// Tokenize + UMT5-encode `prompt` → `[1, 512, 4096]` (bf16, sc-12778), zero-padded to `max_length`
    /// (the DiT cross-attends over the 512-padded context — the same rule as the 5B, sc-3697). Shared
    /// Wan text-encode routine (sc-9000 / F-020).
    fn encode(&self, comps: &Components, prompt: &str) -> CResult<Tensor> {
        self.encode_raw(&comps.tok, &comps.te, prompt)
    }

    /// The tokenizer+encoder core of [`encode`](Self::encode), taking the two text components directly
    /// so the sequential path can drive it with its staged (about-to-be-dropped) UMT5 encoder rather
    /// than the resident [`Components`] bundle (sc-12733). Produces the same raw `[1, 512, 4096]` f32
    /// context either way — the per-expert `embed_text` projection happens later, at the expert.
    fn encode_raw(&self, tok: &TextTokenizer, te: &Umt5Encoder, prompt: &str) -> CResult<Tensor> {
        crate::text_encode::umt5_encode_padded(
            tok,
            &self.te_cfg,
            te,
            prompt,
            &self.device,
            ENC_DTYPE,
            "wan-14b",
        )
    }

    /// Build the I2V channel-concat conditioning `y` `[1, 20, t_lat, h_lat, w_lat]` =
    /// `[mask(4), z_video(16)]`: a conditioning video (frame 0 = the preprocessed image, the rest zero)
    /// is z16-VAE-encoded, and a temporal mask (1.0 at latent frame 0, else 0.0) is prepended. Mirrors
    /// `generate_wan.py`'s `is_i2v_channel_concat` setup. Constant across denoise steps + both experts.
    fn build_i2v_y(
        &self,
        vae: &WanVae16,
        image: &Image,
        frames: u32,
        width: u32,
        height: u32,
    ) -> CResult<Tensor> {
        // Conditioning video [1, 3, F, H, W]: frame 0 = image (in [-1,1]), rest zeros.
        let first = preprocess_i2v_image(image, width, height, &self.device)?; // [1,3,1,H,W]
        let video = if frames > 1 {
            let rest = Tensor::zeros(
                (1, 3, frames as usize - 1, height as usize, width as usize),
                DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[&first, &rest], 2)?
        } else {
            first
        };
        let z_video = vae.encode(&video)?; // [1, 16, t_lat, h_lat, w_lat]

        // Mask dims follow the encoder's actual output, so they always match `z_video`.
        let (_, _, t_lat, h_lat, w_lat) = z_video.dims5()?;
        // 4-channel temporal mask: 1.0 at latent frame 0 (all channels/spatial), 0.0 elsewhere.
        let plane = h_lat * w_lat;
        let mut mask = vec![0f32; 4 * t_lat * plane];
        for c in 0..4 {
            let base = c * t_lat * plane; // temporal index 0 of channel c
            for v in mask.iter_mut().skip(base).take(plane) {
                *v = 1.0;
            }
        }
        let mask = Tensor::from_vec(mask, (1, 4, t_lat, h_lat, w_lat), &self.device)?;
        Ok(Tensor::cat(&[&mask, &z_video], 1)?) // [1, 20, t_lat, h_lat, w_lat]
    }

    /// Resolve the per-request knobs against the variant/config defaults — shared verbatim by the
    /// resident [`render`](Self::render) and the sequential [`render_sequential`](Self::render_sequential)
    /// so the two paths can never resolve steps/guidance/boundary differently (the residency change must
    /// be numerics-preserving, sc-12733).
    fn resolve_knobs(&self, req: &GenerationRequest) -> RenderKnobs {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(DEFAULT_STEPS_14B as usize);
        let (boundary, default_shift, gl, gh) = self.variant.moe_knobs();
        let shift = req
            .scheduler_shift
            .map(|s| s as f64)
            .unwrap_or(default_shift);
        // A scalar request guidance overrides both experts; else the per-expert (low, high) defaults.
        let (g_low, g_high) = match req.guidance {
            Some(g) => (g as f64, g as f64),
            None => (gl as f64, gh as f64),
        };
        RenderKnobs {
            steps,
            frames: req.frames.unwrap_or(DEFAULT_FRAMES_14B),
            fps: req.fps.unwrap_or(DEFAULT_FPS_14B),
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
            sampler: Sampler::parse(req.sampler.as_deref()),
            shift,
            boundary_ts: boundary * NUM_TRAIN_TIMESTEPS as f64,
            g_low,
            g_high,
        }
    }

    /// Latent geometry (z16 strides) + the shared-token-grid RoPE tables `(t_lat, h_lat, w_lat, cos,
    /// sin)`. Shared by both render paths (sc-12733).
    fn geometry(
        &self,
        req: &GenerationRequest,
        frames: u32,
    ) -> CResult<(usize, usize, usize, Tensor, Tensor)> {
        let t_lat = ((frames - 1) / VAE16_STRIDE_TEMPORAL + 1) as usize;
        let h_lat = (req.height / VAE16_STRIDE_SPATIAL) as usize;
        let w_lat = (req.width / VAE16_STRIDE_SPATIAL) as usize;
        let (pt, ph, pw) = self.dit_cfg.patch;
        let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
        let (cos, sin) = WanRope::new(&self.dit_cfg).cos_sin(ppf, pph, ppw, &self.device)?;
        Ok((t_lat, h_lat, w_lat, cos, sin))
    }

    /// The shared UMT5 negative-context prompt when either expert has CFG active — `None` at
    /// guidance ≤ 1.0 (sc-8993, the encode-time gate mirrored from [`cfg_active`]).
    fn negative_prompt<'a>(
        &self,
        req: &'a GenerationRequest,
        knobs: &RenderKnobs,
    ) -> Option<&'a str> {
        if cfg_active(knobs.g_high) || cfg_active(knobs.g_low) {
            Some(req.negative_prompt.as_deref().unwrap_or(NEGATIVE_FALLBACK))
        } else {
            None
        }
    }

    /// Run the denoise steps `range` on a single expert, advancing the (shared, continuous) scheduler
    /// and mutating `latents` in place. Extracted so the resident MoE loop and the staged expert-swap
    /// drive the **identical** per-step math over their respective step ranges — the prefix/suffix split
    /// is bit-exact to the resident per-step `t ≥ boundary` choice because the same `sched` advances
    /// through every step in order (sc-12733).
    #[allow(clippy::too_many_arguments)]
    fn denoise_range(
        &self,
        expert: &WanTransformer,
        ctx_pos: &Tensor,
        ctx_neg: Option<&Tensor>,
        guidance: f64,
        y: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
        sched: &mut FlowScheduler,
        latents: &mut Tensor,
        range: std::ops::Range<usize>,
        total: u32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<()> {
        for i in range {
            check_cancel(cancel)?;
            let t = sched.timestep(i);
            // I2V: concat the conditioning `y` onto the noise latent (→ in_dim 36) before the forward.
            let x = match y {
                Some(y) => Tensor::cat(&[&*latents, y], 1)?,
                None => latents.clone(),
            };
            let v_pos = expert.forward(&x, ctx_pos, t, cos, sin)?;
            // Negative branch (and CFG combine) only when this expert's guidance enables it; `ctx_neg`
            // is `Some` iff that guidance > 1.0 (sc-8993).
            let v = match ctx_neg {
                Some(ctx_neg) if cfg_active(guidance) => {
                    let v_neg = expert.forward(&x, ctx_neg, t, cos, sin)?;
                    cfg(&v_pos, &v_neg, guidance)?
                }
                _ => v_pos,
            };
            *latents = sched.step(&v, latents)?; // 16-channel latent (out_dim 16)
            on_progress(Progress::Step {
                current: i as u32 + 1,
                total,
            });
        }
        Ok(())
    }

    /// The resident MoE render (unchanged residency: `Components` holds UMT5, **both** experts and the
    /// VAE co-resident for the whole generation). Encodes once, projects both experts' contexts, then
    /// splits the denoise at the boundary-crossing index and drives each half through
    /// [`denoise_range`](Self::denoise_range).
    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let knobs = self.resolve_knobs(req);

        // Text encode (pos always) once; project to each expert's context (per-expert text_embedder).
        // The negative branch is only used at guidance > 1.0, and the two experts can have distinct
        // guidance — so UMT5-encode + project the negative for an expert only when its own guidance
        // enables CFG. At guidance <= 1.0 the denoise loop never touches `*_neg`, so the 24-layer UMT5
        // forward over the negative and its projection are pure waste (sc-8993).
        let pos = self.encode(comps, &req.prompt)?;
        let high_pos = comps.high.embed_text(&pos)?;
        let low_pos = comps.low.embed_text(&pos)?;
        // Shared UMT5 negative encode, computed once if either expert has CFG active.
        let neg = match self.negative_prompt(req, &knobs) {
            Some(neg_prompt) => Some(self.encode(comps, neg_prompt)?),
            None => None,
        };
        let high_neg = match &neg {
            Some(neg) if cfg_active(knobs.g_high) => Some(comps.high.embed_text(neg)?),
            _ => None,
        };
        let low_neg = match &neg {
            Some(neg) if cfg_active(knobs.g_low) => Some(comps.low.embed_text(neg)?),
            _ => None,
        };

        let (t_lat, h_lat, w_lat, cos, sin) = self.geometry(req, knobs.frames)?;

        // I2V: build the constant channel-concat conditioning `y` (needs the VAE encoder).
        let y = match self.variant {
            Variant::I2v => {
                let image = i2v_reference(req).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "{}: image-to-video requires a Reference conditioning image",
                        self.variant.id()
                    ))
                })?;
                Some(self.build_i2v_y(&comps.vae, image, knobs.frames, req.width, req.height)?)
            }
            Variant::T2v => None,
        };

        let mut latents = create_noise(knobs.seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let mut sched = FlowScheduler::new(knobs.sampler, knobs.steps, knobs.shift);
        let total = knobs.steps as u32;
        // MoE: high-noise expert at/above the boundary timestep, low-noise below. Flow-match timesteps
        // are monotonically decreasing, so the crossing is a single prefix/suffix split (sc-12733).
        let k = crossing_index(&sched, knobs.steps, knobs.boundary_ts);
        self.denoise_range(
            &comps.high,
            &high_pos,
            high_neg.as_ref(),
            knobs.g_high,
            y.as_ref(),
            &cos,
            &sin,
            &mut sched,
            &mut latents,
            0..k,
            total,
            &req.cancel,
            on_progress,
        )?;
        self.denoise_range(
            &comps.low,
            &low_pos,
            low_neg.as_ref(),
            knobs.g_low,
            y.as_ref(),
            &cos,
            &sin,
            &mut sched,
            &mut latents,
            k..knobs.steps,
            total,
            &req.cancel,
            on_progress,
        )?;

        on_progress(Progress::Decoding);
        // sc-12758: free-aware budgeted **spatial** tiling for the z16 decode. Falls back to plain
        // `decode` when a single high-res frame already fits (behavior-identical when the budget is
        // ample); tiles the 42 GB A14B decode spike down to fit a small card otherwise.
        let decoded = comps
            .vae
            .decode_budgeted_with_cancel(&latents, &req.cancel)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, knobs.fps))
    }

    /// The **sequential-offload** MoE render (sc-12733, epic 12732) — the staged twin of
    /// [`render`](Self::render) that keeps only one heavy component resident at a time, dropping the
    /// pre-decode (denoise) peak on a 24 GB card:
    ///
    /// 1. **TE off-GPU during denoise.** Load UMT5, encode the pos (+ neg when CFG) **raw** `[1,512,4096]`
    ///    context, then DROP the ~11 GB bf16 encoder (sc-12778, was ~21 GB f32) — only the small context
    ///    tensors survive.
    /// 2. **VAE staged (I2V).** Load the VAE (with encoder), build the channel-concat `y`, drop it before
    ///    denoise; the encoder is only needed to build `y`.
    /// 3. **Expert swap.** Hold only the **active** expert: load high, project its context, run steps
    ///    `0..k` (`t ≥ boundary`), DROP high; then load low, project, run `k..steps`. At most one swap
    ///    (one boundary crossing) — the two ~8-9 GB experts are **never co-resident**.
    /// 4. **VAE decode.** Reload the VAE and decode.
    ///
    /// Parity: the per-expert `embed_text` projection runs through **each expert's own** `text_embedder`
    /// (entangled), so the projection must happen after that expert loads — the raw UMT5 context stays
    /// resident across the swap. One [`FlowScheduler`] advances continuously across the swap, so the
    /// prefix/suffix split reproduces the resident per-step choice bit-for-bit. Each heavy component is a
    /// local bound to its own scope, so Rust's scope drop frees the inactive one before the next loads;
    /// a `device.synchronize()` at each boundary drains the async encode/denoise kernels before the freed
    /// pool is reused (the sc-12195 eviction race, applied here per-swap rather than once).
    fn render_sequential(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        let knobs = self.resolve_knobs(req);
        let cancel = &req.cancel;
        check_cancel(cancel)?;

        // The tiny tokenizer is cheap and stays resident across the whole render; the heavy UMT5 encoder
        // is dropped right after encoding.
        let tok = crate::text_encode::build_umt5_tokenizer(&self.root, &self.te_cfg, "wan-14b")?;

        // ── Stage 1: UMT5 text encode → raw pos/neg context → DROP the encoder (~11 GB bf16, sc-12778). ──
        let (pos, neg) = {
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            let te = self.load_te()?;
            let pos = self.encode_raw(&tok, &te, &req.prompt)?;
            let neg = match self.negative_prompt(req, &knobs) {
                Some(neg_prompt) => Some(self.encode_raw(&tok, &te, neg_prompt)?),
                None => None,
            };
            // sc-12195: the encode kernels are async — drain before `te` frees at the brace below, or
            // the next heavy load reuses the freed allocator pool under in-flight kernels.
            self.device.synchronize()?;
            (pos, neg)
        };
        check_cancel(cancel)?;

        let (t_lat, h_lat, w_lat, cos, sin) = self.geometry(req, knobs.frames)?;

        // ── Stage 1b (I2V only): load the VAE (with encoder), build `y`, DROP the VAE before denoise. ──
        let y = match self.variant {
            Variant::I2v => {
                on_progress(Progress::Loading(LoadPhase::Renderer));
                let vae = self.load_vae()?;
                let image = i2v_reference(req).ok_or_else(|| {
                    CandleError::Msg(format!(
                        "{}: image-to-video requires a Reference conditioning image",
                        self.variant.id()
                    ))
                })?;
                let y = self.build_i2v_y(&vae, image, knobs.frames, req.width, req.height)?;
                self.device.synchronize()?; // drain the encode before the VAE frees at the brace.
                Some(y)
            }
            Variant::T2v => None,
        };
        check_cancel(cancel)?;

        let mut latents = create_noise(knobs.seed, Z_DIM, t_lat, h_lat, w_lat, &self.device)?;
        let mut sched = FlowScheduler::new(knobs.sampler, knobs.steps, knobs.shift);
        let total = knobs.steps as u32;
        // The single high→low crossing: high runs `0..k` (`t ≥ boundary`), low runs `k..steps`.
        let k = crossing_index(&sched, knobs.steps, knobs.boundary_ts);

        // ── Stage 2: the expert swap — hold only the ACTIVE expert (the two are NEVER co-resident). ──
        // Load high, project + denoise `0..k`, DROP high; then load low, project + denoise `k..steps`.
        // Driven through [`staged_expert_swap`], whose block-scoping frees the inactive expert before the
        // next loads; the `sync` boundary drains in-flight kernels before the freed pool is reused. The
        // high expert's applied-residual count is carried past its drop in a `Cell` (a `usize`, not the
        // expert) so the cross-expert zero-match guard runs once both have loaded.
        let high_applied: std::cell::Cell<Option<usize>> = std::cell::Cell::new(None);
        let mut swap_state = SwapState {
            sched: &mut sched,
            latents: &mut latents,
            // Reborrow so `on_progress` is usable again for Stage 3 after `swap_state` dies.
            on_progress: &mut *on_progress,
        };
        staged_expert_swap(
            k,
            knobs.steps,
            &mut swap_state,
            // load high
            |st| {
                check_cancel(cancel)?;
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (mut high, applied) = self.load_expert_staged(MoeExpert::High)?;
                // sc-12768: bound the async denoise pipeline (per-block stream drain) so it can't race
                // the churned cudarc pool this staged path reuses — the full-res illegal-memory access.
                high.set_bounded_offload(true);
                high_applied.set(applied);
                Ok(high)
            },
            // use high over `0..k`
            |high, st| {
                let high_pos = high.embed_text(&pos)?;
                let high_neg = match &neg {
                    Some(neg) if cfg_active(knobs.g_high) => Some(high.embed_text(neg)?),
                    _ => None,
                };
                self.denoise_range(
                    high,
                    &high_pos,
                    high_neg.as_ref(),
                    knobs.g_high,
                    y.as_ref(),
                    &cos,
                    &sin,
                    st.sched,
                    st.latents,
                    0..k,
                    total,
                    cancel,
                    st.on_progress,
                )
            },
            // load low
            |st| {
                check_cancel(cancel)?;
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (mut low, low_applied) = self.load_expert_staged(MoeExpert::Low)?;
                // sc-12768: same per-block stream drain as the high expert — this is the expert whose
                // full-res forward faulted at the swap boundary without it.
                low.set_bounded_offload(true);
                // Both experts have now loaded (whenever both own steps) → the cross-expert guard is exact.
                self.adapter_zero_match_guard(high_applied.get(), low_applied)?;
                Ok(low)
            },
            // use low over `k..steps`
            |low, st| {
                let low_pos = low.embed_text(&pos)?;
                let low_neg = match &neg {
                    Some(neg) if cfg_active(knobs.g_low) => Some(low.embed_text(neg)?),
                    _ => None,
                };
                self.denoise_range(
                    low,
                    &low_pos,
                    low_neg.as_ref(),
                    knobs.g_low,
                    y.as_ref(),
                    &cos,
                    &sin,
                    st.sched,
                    st.latents,
                    k..knobs.steps,
                    total,
                    cancel,
                    st.on_progress,
                )
            },
            // sc-12195 boundary sync: drain kernels before the dropped expert's pool is reused.
            || Ok(self.device.synchronize()?),
        )?;
        check_cancel(cancel)?;

        // ── Stage 3: reload the VAE and decode. ──
        on_progress(Progress::Loading(LoadPhase::Renderer));
        let vae = self.load_vae()?;
        on_progress(Progress::Decoding);
        // sc-12758: the experts + TE are offloaded by now, so the decode budgets against nearly the
        // whole card. Free-aware budgeted spatial tiling drives the 42 GB z16 decode spike down to fit
        // the free VRAM — the sole thing that kept the A14B off a 24 GB card (denoise is only ~11 GB).
        let decoded = vae.decode_budgeted_with_cancel(&latents, cancel)?;
        let images = frames_to_images(&decoded)?;
        Ok((images, knobs.fps))
    }
}

/// Resolved per-request render knobs (steps/guidance/boundary/…), produced by
/// [`Pipeline::resolve_knobs`] and consumed identically by the resident and sequential render paths so
/// the residency change stays numerics-preserving (sc-12733).
struct RenderKnobs {
    steps: usize,
    frames: u32,
    fps: u32,
    seed: u64,
    sampler: Sampler,
    shift: f64,
    /// Boundary timestep `boundary · num_train_timesteps` (T2V 875, I2V 900).
    boundary_ts: f64,
    g_low: f64,
    g_high: f64,
}

/// First step index whose flow-match timestep drops **below** `boundary_ts` — the single high→low MoE
/// crossing. Steps `0..k` run the high-noise expert (`t ≥ boundary_ts`), `k..steps` the low-noise
/// expert. Flow-match timesteps are monotonically decreasing, so this prefix/suffix split is exactly the
/// per-step `t ≥ boundary_ts` choice (returns `steps` if the boundary is never crossed → all high; `0`
/// if it is crossed at the first step → all low). Pure and GPU-free, so both render paths share it and a
/// unit test can pin the split against the per-step rule (sc-12733).
fn crossing_index(sched: &FlowScheduler, steps: usize, boundary_ts: f64) -> usize {
    (0..steps)
        .find(|&i| sched.timestep(i) < boundary_ts)
        .unwrap_or(steps)
}

/// The mutable denoise state threaded through [`staged_expert_swap`] (sc-12733). Held as `&mut`
/// references so exclusive access moves between the load/use closures via the `&mut SwapState` param
/// rather than being captured by each closure — the borrow-checker-clean way to let two `FnOnce`
/// stages share the scheduler/latents without a `RefCell`.
struct SwapState<'a> {
    sched: &'a mut FlowScheduler,
    latents: &'a mut Tensor,
    on_progress: &'a mut dyn FnMut(Progress),
}

/// Drive the A14B high→low MoE expert swap so the two ~8-9 GB experts are **never co-resident**
/// (sc-12733, the Pillar-1 win): load the high expert, `use_high` it over steps `0..k`, DROP it, then
/// load the low expert and `use_low` it over `k..steps`. Each expert is a local bound to its own block,
/// so Rust's scope drop frees the inactive one **before** the next loads; an expert whose step range is
/// empty (`k == 0` all-low, or `k == steps` all-high) is skipped entirely — a single boundary crossing,
/// at most one swap. `sync` runs at each swap boundary to drain in-flight kernels before the dropped
/// expert's allocator pool is reused (the sc-12195 eviction race, applied per-swap).
///
/// Generic over the expert type `E` and the threaded state `St` so a CPU unit test can pin the
/// never-co-resident property with a lightweight liveness witness — no GPU, no real weights — exactly as
/// `candle_gen::residency`'s `run_sequential` is pinned. The load closures receive `&mut St` so they can
/// emit their [`Progress::Loading`] before the (heavy) load; the use closures receive `&mut St` to
/// advance the shared scheduler/latents.
#[allow(clippy::too_many_arguments)]
fn staged_expert_swap<E, St>(
    k: usize,
    steps: usize,
    state: &mut St,
    load_high: impl FnOnce(&mut St) -> CResult<E>,
    use_high: impl FnOnce(&E, &mut St) -> CResult<()>,
    load_low: impl FnOnce(&mut St) -> CResult<E>,
    use_low: impl FnOnce(&E, &mut St) -> CResult<()>,
    mut sync: impl FnMut() -> CResult<()>,
) -> CResult<()> {
    if k > 0 {
        let high = load_high(state)?;
        use_high(&high, state)?;
        // Drain before `high` frees at the brace below and `low` reuses the pool.
        sync()?;
    } // `high` drops HERE — freed before `low` is ever loaded (the never-co-resident invariant).
    if k < steps {
        // sc-12768: drain AGAIN now that `high` has dropped its weights back into candle's in-process
        // cudarc caching pool, BEFORE `low` loads and reuses those pages. The free→realloc across the
        // swap must be fully ordered — an un-drained reuse of the churned pool by the low expert's
        // full-res weights/activations faults with a CUDA illegal-memory access at the 720p A14B
        // geometry (`load_low`/`use_low` share this pool with the just-freed high expert).
        if k > 0 {
            sync()?;
        }
        let low = load_low(state)?;
        use_low(&low, state)?;
        sync()?;
    } // `low` drops here.
    Ok(())
}

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

/// Preprocess an I2V conditioning [`Image`] to `[1, 3, 1, height, width]` f32 in `[-1, 1]`: a cover-fit
/// resize (`scale = max(W/iw, H/ih)`) + center-crop to the target, then `px/255·2 − 1`. Uses **bilinear**
/// resampling (the reference's PIL-exact LANCZOS, for bit-exact MLX parity, is a follow-up — sc-5174).
pub fn preprocess_i2v_image(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> CResult<Tensor> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "wan-14b i2v image buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit scale + integer resize dims (≥ target so the center-crop is fully covered).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = ((iw as f64 * scale).round() as usize).max(tw);
    let nh = ((ih as f64 * scale).round() as usize).max(th);
    let resized = bilinear_rgb(&image.pixels, iw, ih, nw, nh);
    // Center-crop to (th, tw), normalize → CHW [-1,1].
    let (x1, y1) = ((nw - tw) / 2, (nh - th) / 2);
    let plane = th * tw;
    let mut chw = vec![0f32; 3 * plane];
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, 1, th, tw), device)?)
}

/// Bilinear resize of an `iw×ih` RGB8 (HWC) buffer to `nw×nh`, returning HWC f32 pixel values in
/// `[0, 255]` (not normalized).
fn bilinear_rgb(px: &[u8], iw: usize, ih: usize, nw: usize, nh: usize) -> Vec<f32> {
    let mut out = vec![0f32; nw * nh * 3];
    let sx = iw as f64 / nw as f64;
    let sy = ih as f64 / nh as f64;
    for oy in 0..nh {
        // Pixel-center mapping (align_corners=False), clamped to the source extent.
        let fy = ((oy as f64 + 0.5) * sy - 0.5).clamp(0.0, (ih - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(ih - 1);
        let wy = fy - y0 as f64;
        for ox in 0..nw {
            let fx = ((ox as f64 + 0.5) * sx - 0.5).clamp(0.0, (iw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(iw - 1);
            let wx = fx - x0 as f64;
            for c in 0..3 {
                let p = |y: usize, x: usize| px[(y * iw + x) * 3 + c] as f64;
                let top = p(y0, x0) * (1.0 - wx) + p(y0, x1) * wx;
                let bot = p(y1, x0) * (1.0 - wx) + p(y1, x1) * wx;
                out[(oy * nw + ox) * 3 + c] = (top * (1.0 - wy) + bot * wy) as f32;
            }
        }
    }
    out
}

/// A loaded Wan2.2 A14B generator (T2V or I2V). Heavy components (UMT5, the two 14B experts, the z16
/// VAE) are loaded lazily on the first `generate` and cached.
pub struct Wan14bGenerator {
    descriptor: ModelDescriptor,
    variant: Variant,
    root: PathBuf,
    device: Device,
    adapters: Vec<AdapterSpec>,
    /// In-place ComfyUI experts (epic 10451 Phase 2c, sc-10671), set only by
    /// [`load_from_comfyui_experts`]. When present, the lazy component build sources both experts from
    /// these files, the UMT5 TE + VAE in place when their files are set (sc-10909) else from
    /// [`Self::root`], and the tiny tokenizer always from [`Self::root`]; `None` on the registry path.
    comfyui: Option<std::sync::Arc<crate::comfyui::ComfyuiExperts>>,
    /// Component-residency policy (epic 12732, sc-12733), resolved once at load via
    /// [`effective_offload_policy`] (honoring both `LoadSpec::offload_policy` and the family-wide
    /// `CANDLE_GEN_OFFLOAD=sequential` A/B override). [`OffloadPolicy::Resident`] keeps the cached
    /// [`Components`] warm; [`OffloadPolicy::Sequential`] drives the staged
    /// [`Pipeline::render_sequential`] (TE-offload + expert-swap + VAE-staging), bounding the denoise
    /// peak on a 24 GB card. The resident [`components`](Self::components) cache stays untouched under
    /// `Sequential` — the staged path never populates it.
    offload: OffloadPolicy,
    components: Mutex<Option<Components>>,
}

impl Wan14bGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `cached` recovers a poisoned lock (sc-9015) internally; `?` bridges the candle-side
        // `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for Wan14bGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.variant.id();
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE_14B)
            || !req.height.is_multiple_of(SIZE_MULTIPLE_14B)
        {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE_14B} (got {}x{})",
                req.width, req.height
            )));
        }
        // The A14B MoE keeps two resident 14B experts; an over-area request is a far-over-envelope run
        // that fails opaquely (OOM). Reject past the documented cap with an actionable message (sc-9028).
        let area = req.width as usize * req.height as usize;
        if area > MAX_AREA_14B {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width×height ({}×{} = {area} px) exceeds the max area {MAX_AREA_14B} px \
                 (1280×720); reduce the resolution",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % 4 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: frames must satisfy frames % 4 == 1 (got {f})"
                )));
            }
            if f as usize > crate::MAX_WAN_FRAMES {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: frames {f} exceeds the maximum {}",
                    crate::MAX_WAN_FRAMES
                )));
            }
        }
        if self.variant == Variant::I2v && i2v_reference(req).is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: image-to-video requires a Reference conditioning image"
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = match &self.comfyui {
            Some(experts) => {
                Pipeline::load_comfyui(&self.root, &self.device, self.variant, experts.clone())
            }
            None => Pipeline::load(
                &self.root,
                &self.device,
                self.variant,
                self.adapters.clone(),
            ),
        };
        // Sequential offload (sc-12733): stage load→use→drop each heavy component so the denoise peak is
        // one expert instead of TE + both experts + VAE co-resident. Resident (default): the cached
        // `Components` bundle, unchanged path. The staged path never populates the resident cache.
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

/// Shared descriptor surface for both A14B variants — CFG (per-expert guidance) + negative prompt,
/// UniPC/Euler samplers; H/W multiple of 16; **LoRA/LoKr supported** (sc-5167 — merged per-expert at
/// load; quant still deferred). `conditioning` differs per variant.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[],
        id: variant.id(),
        family: "wan",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: match variant {
                Variant::T2v => vec![],
                Variant::I2v => vec![ConditioningKind::Reference],
            },
            supports_lora: true,
            supports_lokr: true,
            // Curated `uni_pc` (sc-7296) → Wan's native UniPC; `euler` flow Euler. Legacy `unipc` alias.
            samplers: vec!["uni_pc", "euler", "unipc"],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            // Q4/Q8 packed MLX tiers (sc-10025): both dual-expert `WanTransformer` backbones load packed
            // via the shared packed-detect loaders; the tiers are pre-quantized (no on-the-fly quant).
            // Tier ingestion (MLX layout + key remap) is sc-10026.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12733): the staged
            // `render_sequential` offloads UMT5 during denoise and holds only the ACTIVE MoE expert
            // resident (never both), dropping the pre-decode peak on a 24 GB card. Advertised so the
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

/// Wan2.2 T2V-A14B dual-expert MoE text→video descriptor.
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::T2v)
}

/// Wan2.2 I2V-A14B dual-expert MoE channel-concat image→video descriptor.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    descriptor_for(Variant::I2v)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(build_generator(spec, variant)?))
}

/// The concrete registry-path constructor behind [`load_variant`] — validates the spec surface and
/// resolves the residency policy, returning the concrete [`Wan14bGenerator`] so the offload-policy
/// wiring is unit-testable without a `dyn Generator` downcast (sc-12733).
fn build_generator(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Wan14bGenerator> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ transformer_2/ vae/ \
                 tokenizer/), not a single .safetensors file"
            )));
        }
    };
    // No `spec.quantize` reject (sc-10025): the A14B quant matrix is packed-tier, not on-the-fly — a
    // q4/q8 tier is pre-quantized (the packed-detect loaders read its `.scales`), a dense tier loads
    // dense, so `spec.quantize` is a no-op tier-select marker resolved worker-side (mirrors ltx sc-9417).
    // I2V's conditioning image arrives per-request (`Conditioning::Reference`), not via `spec.control`;
    // the diffusers control/VACE overlays are not wired here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} does not support control / VACE / IP-adapter overlays"
        )));
    }
    let device = candle_gen::default_device()?;
    // Resolve the residency policy once (sc-12733): honors both `spec.offload_policy` and the
    // family-wide `CANDLE_GEN_OFFLOAD=sequential` A/B override.
    let offload = effective_offload_policy(spec.offload_policy);
    Ok(Wan14bGenerator {
        descriptor: descriptor_for(variant),
        variant,
        root,
        device,
        adapters: spec.adapters.clone(),
        comfyui: None,
        offload,
        components: Mutex::new(None),
    })
}

/// Construct a lazy candle Wan2.2 A14B generator that reads its **two DiT experts in place** from an
/// existing ComfyUI install (epic 10451 Phase 2c, sc-10671) — no copy, no re-download. `high_file` /
/// `low_file` are the user's ComfyUI high/low-noise expert files (native-Wan keys, companion scaled-fp8);
/// each is remapped + dequant'd to bf16 in memory (`crate::comfyui`) at component build.
///
/// `te_file` / `vae_file` optionally read the UMT5 text encoder + Wan VAE in place too (sc-10909): the
/// UMT5 (`umt5_xxl_fp8_e4m3fn_scaled`) is the same scaled-fp8 convention (dequant, no key remap), and
/// the VAE (`wan_2.1_vae.safetensors`) is native-WAN-VAE keys remapped to diffusers. When either is
/// `None` that component falls back to the `snapshot_dir` tier. `snapshot_dir` is a resident Wan2.2
/// A14B snapshot tier that always supplies at least the tiny UMT5 tokenizer (and the TE/VAE when their
/// files are absent). `variant` selects the T2V or I2V config (`patch_embedding` in-channels differ).
/// No adapters / control on this lane.
pub fn load_from_comfyui_experts(
    high_file: impl Into<PathBuf>,
    low_file: impl Into<PathBuf>,
    te_file: Option<PathBuf>,
    vae_file: Option<PathBuf>,
    snapshot_dir: impl Into<PathBuf>,
    i2v: bool,
) -> gen_core::Result<Box<dyn Generator>> {
    let variant = if i2v { Variant::I2v } else { Variant::T2v };
    let device = candle_gen::default_device()?;
    // The ComfyUI lane carries no `LoadSpec`, so the residency policy comes purely from the family-wide
    // `CANDLE_GEN_OFFLOAD=sequential` A/B override (default resident) — sc-12733.
    let offload = effective_offload_policy(OffloadPolicy::Resident);
    Ok(Box::new(Wan14bGenerator {
        descriptor: descriptor_for(variant),
        variant,
        root: snapshot_dir.into(),
        device,
        adapters: Vec::new(),
        comfyui: Some(std::sync::Arc::new(crate::comfyui::ComfyuiExperts {
            high_file: high_file.into(),
            low_file: low_file.into(),
            te_file,
            vae_file,
        })),
        offload,
        components: Mutex::new(None),
    }))
}

/// Construct a lazy candle Wan2.2 T2V-A14B generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `Wan-AI/Wan2.2-T2V-A14B-Diffusers` snapshot (`text_encoder/`, `transformer/`,
/// `transformer_2/`, `vae/`, `tokenizer/`).
pub fn load_t2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::T2v)
}

/// Construct a lazy candle Wan2.2 I2V-A14B generator (channel-concat image→video). Same snapshot layout
/// as the T2V variant; the conditioning image arrives per-request as a `Conditioning::Reference`.
pub fn load_i2v_14b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::I2v)
}

candle_gen::register_generators! {
    pub(crate) const T2V_14B_REGISTRATION = descriptor_t2v_14b => load_t2v_14b
}
candle_gen::register_generators! {
    pub(crate) const I2V_14B_REGISTRATION = descriptor_i2v_14b => load_i2v_14b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_both_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for (id, conditioning_len) in [(MODEL_ID_T2V_14B, 0usize), (MODEL_ID_I2V_14B, 1)] {
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .expect("14b model is registered");
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "wan");
            assert_eq!(g.descriptor().backend, "candle");
            assert_eq!(g.descriptor().modality, Modality::Video);
            assert!(!g.descriptor().capabilities.mac_only);
            assert_eq!(
                g.descriptor().capabilities.conditioning.len(),
                conditioning_len
            );
        }
    }

    /// sc-8993: `cfg_active` is the single predicate gating the negative branch. CFG only affects the
    /// output at guidance > 1.0; at 1.0 the combine reduces to `pos` exactly and below 1.0 it's off, so
    /// both the encode-time and per-step negative work must be skipped. Defaults (3.0–4.0) keep it on.
    #[test]
    fn cfg_active_gates_negative_branch() {
        assert!(
            !cfg_active(1.0),
            "guidance 1.0 disables CFG (combine == pos)"
        );
        assert!(!cfg_active(0.0));
        assert!(!cfg_active(0.9));
        assert!(cfg_active(1.0001));
        assert!(cfg_active(3.0), "T2V low default keeps CFG on");
        assert!(cfg_active(4.0), "T2V high default keeps CFG on");
        // Per-expert independence: a mixed (low off / high on) request encodes+projects only the high
        // expert's negative, and vice-versa — mirroring the render's per-expert gating.
        let (g_low, g_high) = (1.0_f64, 4.0_f64);
        let neg_needed = cfg_active(g_low) || cfg_active(g_high);
        assert!(
            neg_needed,
            "shared UMT5 encode runs when either expert needs it"
        );
        assert!(
            !cfg_active(g_low),
            "low expert skips its negative projection"
        );
        assert!(
            cfg_active(g_high),
            "high expert keeps its negative projection"
        );
        // Both off: no negative work at all.
        assert!(!(cfg_active(1.0) || cfg_active(0.5)));
    }

    #[test]
    fn descriptor_surface() {
        let t2v = descriptor_t2v_14b();
        assert!(t2v.capabilities.supports_guidance);
        assert!(t2v.capabilities.supports_negative_prompt);
        assert!(!t2v.capabilities.supports_true_cfg);
        assert!(t2v.capabilities.conditioning.is_empty());
        assert!(t2v.capabilities.samplers.contains(&"uni_pc")); // curated spelling (sc-7296)
        assert!(t2v.capabilities.samplers.contains(&"unipc")); // legacy alias retained

        let i2v = descriptor_i2v_14b();
        assert!(i2v.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_enforces_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t2v = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_T2V_14B, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            guidance: Some(4.0),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(t2v.validate(&ok).is_ok());
        assert!(t2v
            .validate(&GenerationRequest {
                frames: Some(1025),
                ..ok.clone()
            })
            .is_ok());
        let over = t2v
            .validate(&GenerationRequest {
                frames: Some(1029),
                ..ok.clone()
            })
            .expect_err("1029 must exceed the Wan frame ceiling");
        assert!(over.to_string().contains("maximum 1025"), "{over}");
        // Legacy `unipc` spelling stays accepted (sc-7296 alias).
        assert!(t2v
            .validate(&GenerationRequest {
                sampler: Some("unipc".into()),
                ..ok.clone()
            })
            .is_ok());
        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // frames not ≡ 1 (mod 4)
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(16),
                ..Default::default()
            },
            // size not a multiple of 16
            GenerationRequest {
                prompt: "x".into(),
                width: 300,
                ..Default::default()
            },
            // unadvertised sampler
            GenerationRequest {
                prompt: "x".into(),
                sampler: Some("dpmpp2m".into()),
                ..Default::default()
            },
            // over the MAX_AREA_14B envelope — 1280×1280 (both grid-aligned) is 1.8× the cap (sc-9028)
            GenerationRequest {
                prompt: "x".into(),
                width: 1280,
                height: 1280,
                frames: Some(17),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
        ] {
            assert!(t2v.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // I2V rejects a request with no Reference image.
        let i2v = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_I2V_14B, &spec)
            .unwrap();
        assert!(i2v.validate(&ok).is_err(), "i2v needs a reference image");
        let reference = Conditioning::Reference {
            image: Image {
                width: 16,
                height: 16,
                pixels: vec![0; 16 * 16 * 3],
            },
            strength: None,
        };
        let i2v_at_cap = GenerationRequest {
            conditioning: vec![reference],
            frames: Some(1025),
            ..ok.clone()
        };
        assert!(i2v.validate(&i2v_at_cap).is_ok());
        let over = i2v
            .validate(&GenerationRequest {
                frames: Some(1029),
                ..i2v_at_cap
            })
            .expect_err("1029 must exceed the Wan frame ceiling");
        assert!(over.to_string().contains("maximum 1025"), "{over}");
    }

    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_T2V_14B, &spec)
            .unwrap();
        let req = GenerationRequest {
            prompt: " \t\n ".into(),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&req).is_err());
    }

    /// The documented `MAX_AREA_14B` cap is actually enforced: an at-cap request passes and a
    /// grid-aligned over-cap request is rejected with an actionable message (sc-9028 / F-044).
    #[test]
    fn validate_enforces_max_area() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t2v = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_T2V_14B, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };

        // Exactly at the cap (1280×720 = 921 600 px, both multiples of 16) is accepted.
        assert_eq!(1280 * 720, MAX_AREA_14B);
        assert!(t2v
            .validate(&GenerationRequest {
                width: 1280,
                height: 720,
                ..base.clone()
            })
            .is_ok());

        // sc-12308 regression: `1280×720` is the 14B family's CANONICAL 720p (upstream
        // `SUPPORTED_SIZES[t2v-A14B]`), and 720 = 45·16 is on the family's grid. It was rejected
        // while this cap wrongly carried the TI2V-5B's `MAX_AREA_5B` (901 120) — a 5B number whose
        // 704 comes from the 5B's 32-px grid. Both orientations must validate.
        assert_ne!(
            MAX_AREA_14B,
            crate::config::MAX_AREA_5B,
            "the 14B family must not reuse the 5B's area budget (sc-12308)"
        );
        assert!(t2v
            .validate(&GenerationRequest {
                width: 720,
                height: 1280,
                ..base.clone()
            })
            .is_ok());

        // Over the cap while both edges stay within the per-edge range (1280×1024 = 1 310 720 px,
        // both grid-aligned and ≤ 1280) is rejected specifically by the area check, with an
        // actionable message that names the cap.
        let err = t2v
            .validate(&GenerationRequest {
                width: 1280,
                height: 1024,
                ..base.clone()
            })
            .expect_err("over-area request must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max area"), "actionable message: {msg}");

        // The same cap applies to the I2V variant (both keep two resident 14B experts).
        let i2v = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_I2V_14B, &spec)
            .unwrap();
        assert!(
            i2v.validate(&GenerationRequest {
                width: 1280,
                height: 1024,
                ..base
            })
            .is_err(),
            "i2v enforces the same max-area cap"
        );
    }

    #[test]
    fn load_accepts_adapters_and_quant() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // LoRA/LoKr are supported (sc-5167) — load is lazy, so attaching adapters resolves OK
        // (the merge happens at the first `generate`).
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load_t2v_14b(&lora).is_ok());
        // Quant is now a no-op tier-select marker (packed-detect load, sc-10025) — a q4/q8 A14B tier is
        // pre-quantized, so `spec.quantize` no longer rejects; both experts load packed at ingestion.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(load_i2v_14b(&quant).is_ok());
    }

    // ── sc-12733: sequential component offload + expert swap (Pillar 1) ──

    /// Both A14B variants must now advertise `supports_sequential_offload` so the worker's fit-gate can
    /// tell "the staged path bounds peak VRAM here" from a no-op fallback (sc-11126 contract).
    #[test]
    fn descriptor_advertises_sequential_offload() {
        for d in [descriptor_t2v_14b(), descriptor_i2v_14b()] {
            assert!(
                d.capabilities.supports_sequential_offload,
                "A14B must advertise sequential offload (sc-12733)"
            );
        }
    }

    /// The load path resolves the residency policy from `LoadSpec::offload_policy` via
    /// [`effective_offload_policy`]: the default spec stays `Resident` (cached-components, unchanged
    /// path), an explicit `Sequential` spec flips the generator onto the staged expert-swap render.
    #[test]
    fn load_resolves_offload_policy_from_spec() {
        let resident = build_generator(
            &LoadSpec::new(WeightsSource::Dir("/snap".into())),
            Variant::T2v,
        )
        .unwrap();
        assert_eq!(resident.offload, OffloadPolicy::Resident);

        let sequential = build_generator(
            &LoadSpec::new(WeightsSource::Dir("/snap".into()))
                .with_offload_policy(OffloadPolicy::Sequential),
            Variant::I2v,
        )
        .unwrap();
        assert_eq!(sequential.offload, OffloadPolicy::Sequential);
    }

    /// Parity guard for the residency change (sc-12733): the precomputed boundary-crossing index `k`
    /// reproduces the resident per-step `t ≥ boundary_ts` expert choice **exactly**. Flow-match
    /// timesteps are monotonically decreasing, so steps `0..k` (high) and `k..steps` (low) are the same
    /// split the resident loop makes step-by-step — the whole justification for driving the swap as one
    /// continuous scheduler. Checked against the real `FlowScheduler` sigma table for both samplers and
    /// both variants' boundaries.
    #[test]
    fn crossing_index_matches_the_per_step_boundary_choice() {
        let steps = 20;
        for sampler in [Sampler::UniPC, Sampler::Euler] {
            let sched = FlowScheduler::new(sampler, steps, T2V_14B_FLOW_SHIFT);
            // Precondition: strictly decreasing timesteps (else the prefix/suffix split is unsound).
            for i in 1..steps {
                assert!(
                    sched.timestep(i) < sched.timestep(i - 1),
                    "flow-match timesteps must be monotonically decreasing"
                );
            }
            for boundary in [T2V_14B_BOUNDARY, I2V_14B_BOUNDARY] {
                let boundary_ts = boundary * NUM_TRAIN_TIMESTEPS as f64;
                let k = crossing_index(&sched, steps, boundary_ts);
                for i in 0..steps {
                    assert_eq!(
                        i < k,
                        sched.timestep(i) >= boundary_ts,
                        "step {i}: prefix/suffix split must equal the per-step `t >= boundary_ts` \
                         choice (k={k}, boundary_ts={boundary_ts})"
                    );
                }
            }
        }
    }

    /// The boundary extremes collapse the swap to a single expert (never load one that owns no steps):
    /// a boundary above every timestep ⇒ all-low (`k == 0`, high skipped); below every timestep ⇒
    /// all-high (`k == steps`, low skipped).
    #[test]
    fn crossing_index_handles_boundary_extremes() {
        let steps = 12;
        let sched = FlowScheduler::new(Sampler::UniPC, steps, T2V_14B_FLOW_SHIFT);
        assert_eq!(
            crossing_index(&sched, steps, f64::INFINITY),
            0,
            "a boundary above every timestep ⇒ all-low (k == 0)"
        );
        assert_eq!(
            crossing_index(&sched, steps, f64::NEG_INFINITY),
            steps,
            "a boundary below every timestep ⇒ all-high (k == steps)"
        );
    }

    /// A liveness witness for the expert-swap residency tests, mirroring the drop-order witnesses in
    /// `candle_gen::residency`'s tests: it bumps a shared live-counter on construction and drops it on
    /// `Drop`, recording the peak concurrency and an ordered load/drop log.
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

    /// Stands in for a loaded 14B expert: its lifetime on the live-counter is exactly the expert's GPU
    /// residency window in `staged_expert_swap`.
    struct ExpertWitness<'a> {
        tracker: &'a LiveTracker,
        drop_tag: &'static str,
    }

    impl<'a> ExpertWitness<'a> {
        fn new(tracker: &'a LiveTracker, born_tag: &'static str, drop_tag: &'static str) -> Self {
            tracker.born(born_tag);
            Self { tracker, drop_tag }
        }
    }

    impl Drop for ExpertWitness<'_> {
        fn drop(&mut self) {
            self.tracker.died(self.drop_tag);
        }
    }

    /// The Pillar-1 invariant (sc-12733): the two experts are **never co-resident**. Driven through the
    /// production `staged_expert_swap` with `0 < k < steps` (a real swap), the peak live-expert count is
    /// 1 and the high expert drops before the low expert loads — a drop-order witness, not a VRAM read
    /// (candle's cudarc pool makes `nvidia-smi` blind to the drop, so residency is asserted structurally).
    #[test]
    fn expert_swap_is_never_co_resident_and_high_drops_before_low_loads() {
        let tracker = LiveTracker::new();
        let mut state = ();
        let out = staged_expert_swap(
            3, // k: 0 < k < steps → both experts own steps → a genuine swap
            8, // steps
            &mut state,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || Ok(()),
        );
        assert!(out.is_ok());
        assert_eq!(
            tracker.peak.get(),
            1,
            "the two 14B experts must NEVER be co-resident (the whole Pillar-1 win)"
        );
        assert_eq!(
            *tracker.log.borrow(),
            vec!["load-high", "drop-high", "load-low", "drop-low"],
            "the high expert must be dropped before the low expert loads"
        );
    }

    /// Mutation-check (sc-12733 acceptance): force both experts resident and confirm the
    /// never-co-resident assertion regresses — proving the passing test above is not a default-value
    /// false green. This is the exact both-resident bug the story removes (the old `Components` held both
    /// experts for the whole render): binding `high` and `low` in one scope co-resides them, and the SAME
    /// liveness witness the passing test relies on now reports peak concurrency 2, so its `peak == 1`
    /// assertion goes RED.
    #[test]
    fn forcing_both_experts_resident_regresses_the_never_co_resident_assertion() {
        let tracker = LiveTracker::new();
        {
            // MUTATION: the inactive expert is NOT dropped before the next loads.
            let _high = ExpertWitness::new(&tracker, "load-high", "drop-high");
            let _low = ExpertWitness::new(&tracker, "load-low", "drop-low");
            tracker.note("both-resident-denoise");
        }
        assert_eq!(
            tracker.peak.get(),
            2,
            "the forced-both-resident mutation co-resides the two experts"
        );
        assert!(
            tracker.peak.get() > 1,
            "the never-co-resident assertion (peak == 1) MUST fail under the both-resident mutation — \
             the passing test genuinely discriminates co-residence, it is not a false green"
        );
    }

    /// `staged_expert_swap` skips loading the expert whose step range is empty (memory-optimal single
    /// crossing): `k == 0` loads only the low expert, `k == steps` loads only the high expert.
    #[test]
    fn expert_swap_skips_the_expert_that_owns_no_steps() {
        // k == 0 → all-low: the high loader is never called.
        let low_only = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            0,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&low_only, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&low_only, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            *low_only.log.borrow(),
            vec!["load-low", "drop-low"],
            "k == 0 must load ONLY the low expert"
        );

        // k == steps → all-high: the low loader is never called.
        let high_only = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            8,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&high_only, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Ok(ExpertWitness::new(&high_only, "load-low", "drop-low")),
            |_w, _st| Ok(()),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            *high_only.log.borrow(),
            vec!["load-high", "drop-high"],
            "k == steps must load ONLY the high expert"
        );
    }

    /// The sc-12195 eviction sync, applied per-swap (sc-12733), plus the sc-12768 **post-drop** sync: the
    /// boundary `sync` runs after each expert is used and **before** it drops, AND again after the high
    /// expert has dropped and **before** the low expert loads — so the churned cudarc caching pool's
    /// free→realloc is fully ordered before the low expert reuses it (the full-res illegal-memory-access
    /// fix). Mirrors residency.rs's boundary-sync ordering witness.
    #[test]
    fn expert_swap_syncs_before_each_expert_drops() {
        let tracker = LiveTracker::new();
        let mut st = ();
        staged_expert_swap(
            3,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| {
                tracker.note("use-high");
                Ok(())
            },
            |_st| Ok(ExpertWitness::new(&tracker, "load-low", "drop-low")),
            |_w, _st| {
                tracker.note("use-low");
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
                "load-high",
                "use-high",
                "sync",
                "drop-high", //
                "sync",      // sc-12768: drain the churned pool AFTER high drops, before low loads
                "load-low",
                "use-low",
                "sync",
                "drop-low",
            ],
            "each expert is synced before it drops; the churned pool is drained again before low loads"
        );
    }

    /// A load failure on the low expert still drops the (already-used-and-synced) high expert via scope
    /// drop on the `?` path — no leak, and the error propagates.
    #[test]
    fn expert_swap_propagates_a_low_load_failure_after_dropping_high() {
        let tracker = LiveTracker::new();
        let mut st = ();
        let out: CResult<()> = staged_expert_swap(
            3,
            8,
            &mut st,
            |_st| Ok(ExpertWitness::new(&tracker, "load-high", "drop-high")),
            |_w, _st| Ok(()),
            |_st| Err(CandleError::Msg("low expert OOM".into())),
            |_w: &ExpertWitness, _st| Ok(()),
            || Ok(()),
        );
        assert!(matches!(out, Err(CandleError::Msg(_))));
        assert_eq!(
            *tracker.log.borrow(),
            vec!["load-high", "drop-high"],
            "high must have dropped before the low load was even attempted"
        );
        assert_eq!(tracker.peak.get(), 1);
    }
}
