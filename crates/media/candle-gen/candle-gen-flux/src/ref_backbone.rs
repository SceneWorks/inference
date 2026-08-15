//! A tier-detecting FLUX.1 backbone for the **reference** lanes (sc-10103, epic 9083) — the candle
//! twin of `mlx-gen-flux`'s `load_flux1`, which the MLX PuLID / IP-adapter providers delegate their
//! FLUX backbone to.
//!
//! The txt2img generator ([`crate::pipeline`]) already auto-detects the tier — a dense **BFL** snapshot
//! (`flux1-*.safetensors` + `ae.safetensors` at the root) vs a pre-quantized **diffusers-layout MLX
//! turnkey** (`SceneWorks/flux1-dev-mlx` q4/q8/bf16: `transformer/` + `text_encoder{,_2}/` + `vae/`) —
//! and builds the right CLIP / T5 / DiT / VAE for each (`Pipeline::load_components`). But that path was
//! wired only into `load_dev`/`load_schnell`; the reference providers (`candle-gen-pulid`, the FLUX
//! IP-adapter) built their backbone by hand from the single-file BFL layout and so could NOT read the
//! turnkey tiers.
//!
//! [`FluxRefBackbone`] closes that gap by **reusing the exact same** [`Pipeline::load_components`]
//! detect-and-load path the shipped txt2img generator uses, then exposing the three ops a reference lane
//! needs on top of it — [`encode_text`](FluxRefBackbone::encode_text),
//! [`forward_injected`](FluxRefBackbone::forward_injected) (the post-block [`DitImageInjector`] seam,
//! now on both the BFL [`IpFlux`](crate::ip_dit::IpFlux) and the diffusers
//! [`PackedFluxDit`](crate::packed_dit::PackedFluxDit)), and [`decode`](FluxRefBackbone::decode). So a
//! reference lane inherits whatever tier handling the base generator has (q4/q8/bf16), with one
//! packed-detect path and zero drift.
//!
//! PiD is **not** owned here: the PiD super-resolving decoder is a per-generation choice the reference
//! provider builds separately (PuLID's `with_pid`), so [`decode`](FluxRefBackbone::decode) takes the
//! decoder explicitly. The backbone loads with no PiD spec.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::{AdapterSpec, CancelFlag, GenerationMemory, Image};
use candle_gen::Result;
use candle_gen_pid::PidDecoder;

use crate::ip_adapter::FluxIpInjector;
use crate::ip_dit::DitImageInjector;
use crate::pipeline::{Components, Pipeline, SeqHeavy};
use crate::Variant;

/// A loaded, tier-detected FLUX.1 backbone (CLIP + T5 + DiT + VAE + tokenizers) for the reference
/// lanes. Holds the light `Pipeline` handle (variant/root/device/dtype) plus the loaded
/// `Components` — the SAME pair the txt2img generator caches — so every op delegates to the shared,
/// already-validated pipeline code.
pub struct FluxRefBackbone {
    pipeline: Pipeline,
    components: Option<Components>,
    memory: GenerationMemory,
}

/// Opaque request-scoped DiT/VAE phase for a staged reference provider. Public only so the separate
/// `candle-gen-pulid` crate can carry it between text encode, denoise, and decode without depending on
/// FLUX's private tier-specific component enums.
pub struct FluxRefHeavy {
    heavy: SeqHeavy,
    device: Device,
}

impl Drop for FluxRefHeavy {
    fn drop(&mut self) {
        // Every success/error/cancel path synchronizes queued CUDA work before the phase fields are
        // dropped. A primary render error is already in flight here, so cleanup failure is deliberately
        // secondary and must not replace it.
        let _ = self.device.synchronize();
    }
}

impl FluxRefBackbone {
    pub fn validate_native_vae_request(&self, use_pid: bool, cancel: &CancelFlag) -> Result<()> {
        candle_gen::check_cancel(cancel)?;
        if use_pid
            && (self.memory.tile_vae_decode
                || self.memory.chunk_attention
                || self.memory.stream_transformer_blocks)
        {
            return Err(candle_gen::CandleError::Msg(
                "FLUX optimized native-VAE memory rungs do not support PiD".into(),
            ));
        }
        Ok(())
    }

    /// Load the FLUX backbone from a snapshot `root`, auto-detecting the tier: a dense BFL snapshot vs a
    /// pre-quantized diffusers-layout MLX turnkey tier subdir (`…/q4`, `…/q8`, `…/bf16`). `root` is the
    /// tier subdir the worker resolved (via `standard_tier_subdir`) for the packed turnkey, or the plain
    /// BFL snapshot root. `variant` is always [`Variant::Dev`] for PuLID; both variants are supported.
    /// No PiD spec is captured — the reference provider owns its PiD decoder separately (see
    /// [`decode`](Self::decode)).
    pub fn load(root: &Path, variant: Variant, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_with_memory(
            root,
            variant,
            device,
            dtype,
            GenerationMemory::default(),
            Vec::new(),
        )
    }

    /// Load resident components for the baseline, or retain only the lightweight snapshot handle when
    /// staged residency is selected. The latter loads text encoders for encode, releases them, then
    /// loads the DiT/VAE phase (optionally streamed/CPU-decoded) per request.
    pub fn load_with_memory(
        root: &Path,
        variant: Variant,
        device: &Device,
        dtype: DType,
        memory: GenerationMemory,
        adapters: Vec<AdapterSpec>,
    ) -> Result<Self> {
        let optimized =
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks;
        if optimized && !memory.stage_residency {
            return Err(candle_gen::CandleError::Msg(
                "FLUX reference memory rungs require staged residency".into(),
            ));
        }
        let pipeline = Pipeline::load(variant, root, device, dtype, None, adapters);
        let components = if memory.stage_residency {
            None
        } else {
            Some(pipeline.load_components()?)
        };
        Ok(Self {
            pipeline,
            components,
            memory,
        })
    }

    /// Encode `prompt` into FLUX's two conditioning tensors — the T5 sequence `(1, L, 4096)` and the
    /// CLIP pooled vector `(1, 768)` — at the compute dtype. Tier-agnostic: delegates to the shared
    /// `Pipeline::text_embeddings`, which runs the stock or packed text encoders as loaded.
    pub fn encode_text(&self, prompt: &str) -> Result<(Tensor, Tensor)> {
        self.encode_text_with_memory(prompt, &CancelFlag::default())
    }

    pub fn encode_text_with_memory(
        &self,
        prompt: &str,
        cancel: &CancelFlag,
    ) -> Result<(Tensor, Tensor)> {
        candle_gen::check_cancel(cancel)?;
        if let Some(components) = &self.components {
            let encoded = self.pipeline.text_embeddings(components, prompt)?;
            candle_gen::check_cancel(cancel)?;
            return Ok(encoded);
        }
        let encoders = self.pipeline.load_text_residency()?;
        let encoded = (|| {
            candle_gen::check_cancel(cancel)?;
            let encoded = self.pipeline.encode_residency(&encoders, prompt)?;
            candle_gen::check_cancel(cancel)?;
            Ok(encoded)
        })();
        let encoded = candle_gen::synchronize_result(self.pipeline.device(), encoded);
        drop(encoders);
        encoded
    }

    /// Load the request-scoped DiT/VAE phase after text encoding. Resident backbones return `None`;
    /// staged backbones return an opaque heavy phase that must be retained through decode and then
    /// dropped by the caller.
    pub fn load_heavy(&self, cancel: &CancelFlag) -> Result<Option<FluxRefHeavy>> {
        candle_gen::check_cancel(cancel)?;
        if self.components.is_some() {
            return Ok(None);
        }
        let loaded = self.pipeline.load_heavy_residency_with_memory(
            false,
            self.memory.stream_transformer_blocks,
            self.memory.tile_vae_decode,
            cancel,
        );
        // A failed staged load may already have queued CUDA copies for the DiT before VAE/PiD or a
        // cancellation check fails. Synchronize that partial phase before its local components drop;
        // on success this also establishes a clean load/forward boundary for the request-scoped owner.
        let heavy = candle_gen::synchronize_result(self.pipeline.device(), loaded)?;
        Ok(Some(FluxRefHeavy {
            heavy,
            device: self.pipeline.device().clone(),
        }))
    }

    /// The FLUX DiT velocity forward with the optional **post-block** image-stream residual injector —
    /// the PuLID id cross-attn seam. Dispatches to the loaded tier's DiT: the BFL
    /// [`IpFlux::forward_injected`](crate::ip_dit::IpFlux::forward_injected) (dense snapshot) or the
    /// diffusers `PackedFluxDit::forward_injected`
    /// (packed/dense turnkey tier). `injector = None` is the plain FLUX forward. `guidance` is the dev
    /// per-batch embedded guidance (`None` for schnell). The two DiTs take the same argument shapes and
    /// inject at the same layout-agnostic block indices, so the caller's [`DitImageInjector`] is
    /// unchanged across tiers.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
    ) -> Result<Tensor> {
        self.forward_injected_with_memory(
            None,
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance,
            injector,
            &CancelFlag::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_injected_with_memory(
        &self,
        heavy: Option<&FluxRefHeavy>,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        let budget = if self.memory.chunk_attention {
            self.memory
                .attention_chunk_size
                .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE) as u64
        } else {
            candle_gen::ATTN_SCORES_BUDGET as u64
        };
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, false))
            .with_cancel(cancel);
        let window = self
            .memory
            .transformer_window_size
            .map(|value| value as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        if let Some(heavy) = heavy {
            return self.pipeline.forward_injected_residency(
                &heavy.heavy,
                img,
                img_ids,
                txt,
                txt_ids,
                timesteps,
                y,
                guidance,
                injector,
                plan,
                window,
                cancel,
            );
        }
        let components = self.components.as_ref().ok_or_else(|| {
            candle_gen::CandleError::Msg(
                "FLUX staged reference forward requires its heavy phase".into(),
            )
        })?;
        let out = match components {
            Components::Stock { transformer, .. } => transformer.forward_injected_with_memory(
                img, img_ids, txt, txt_ids, timesteps, y, guidance, injector, plan, window, cancel,
            )?,
            Components::Packed { transformer, .. } => transformer.forward_injected_with_memory(
                img, img_ids, txt, txt_ids, timesteps, y, guidance, injector, plan, window, cancel,
            )?,
        };
        Ok(out)
    }

    /// XLabs IP-Adapter forward over either resident or request-scoped heavy components. Both BFL and
    /// packed diffusers layouts expose the same post-QKNorm, pre-RoPE image-query seam.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_ip_with_memory(
        &self,
        heavy: Option<&FluxRefHeavy>,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        pooled: &Tensor,
        guidance: Option<&Tensor>,
        injector: &FluxIpInjector<'_>,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        let budget = if self.memory.chunk_attention {
            self.memory
                .attention_chunk_size
                .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE) as u64
        } else {
            candle_gen::ATTN_SCORES_BUDGET as u64
        };
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, false))
            .with_cancel(cancel);
        let window = self
            .memory
            .transformer_window_size
            .map(|value| value as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        if let Some(heavy) = heavy {
            return self.pipeline.forward_ip_residency(
                &heavy.heavy,
                img,
                img_ids,
                txt,
                txt_ids,
                timesteps,
                pooled,
                guidance,
                injector,
                plan,
                window,
                cancel,
            );
        }
        let components = self.components.as_ref().ok_or_else(|| {
            candle_gen::CandleError::Msg("FLUX staged IP forward requires its heavy phase".into())
        })?;
        match components {
            Components::Stock { transformer, .. } => transformer.forward_with_memory(
                img,
                img_ids,
                txt,
                txt_ids,
                timesteps,
                pooled,
                guidance,
                Some(injector),
                plan,
                window,
                cancel,
            ),
            Components::Packed { transformer, .. } => transformer.forward_ip_with_memory(
                img, img_ids, txt, txt_ids, timesteps, pooled, guidance, injector, plan, window,
                cancel,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_control_with_memory(
        &self,
        heavy: Option<&FluxRefHeavy>,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        pooled: &Tensor,
        guidance: Option<&Tensor>,
        injector: Option<&dyn DitImageInjector>,
        control: Option<(&[Tensor], f64)>,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        let budget = if self.memory.chunk_attention {
            self.memory
                .attention_chunk_size
                .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE) as u64
        } else {
            candle_gen::ATTN_SCORES_BUDGET as u64
        };
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(budget, false))
            .with_cancel(cancel);
        let window = self
            .memory
            .transformer_window_size
            .map(|value| value as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        if let Some(heavy) = heavy {
            return self.pipeline.forward_control_residency(
                &heavy.heavy,
                img,
                img_ids,
                txt,
                txt_ids,
                timesteps,
                pooled,
                guidance,
                injector,
                control,
                plan,
                window,
                cancel,
            );
        }
        let components = self.components.as_ref().ok_or_else(|| {
            candle_gen::CandleError::Msg(
                "FLUX staged control forward requires its heavy phase".into(),
            )
        })?;
        match components {
            Components::Stock { transformer, .. } => transformer.forward_control_with_memory(
                img, img_ids, txt, txt_ids, timesteps, pooled, guidance, injector, control, plan,
                window, cancel,
            ),
            Components::Packed { transformer, .. } => transformer.forward_control_with_memory(
                img, img_ids, txt, txt_ids, timesteps, pooled, guidance, injector, control, plan,
                window, cancel,
            ),
        }
    }

    pub fn encode_control_with_memory(
        &self,
        heavy: Option<&FluxRefHeavy>,
        image: &Tensor,
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        candle_gen::check_cancel(cancel)?;
        if let Some(heavy) = heavy {
            return self
                .pipeline
                .encode_control_ref_residency(&heavy.heavy, image, cancel);
        }
        let components = self.components.as_ref().ok_or_else(|| {
            candle_gen::CandleError::Msg(
                "FLUX staged control encode requires its heavy phase".into(),
            )
        })?;
        let encoded = self.pipeline.encode_control_ref(components, image)?;
        candle_gen::check_cancel(cancel)?;
        Ok(encoded)
    }

    /// Decode the denoised latents `(1, h·w, 64)` to an RGB8 [`Image`], routing through the loaded tier's
    /// VAE (stock `AutoEncoder` / packed `AutoEncoderKL`) — or, when `pid` is `Some`, the caller's PiD
    /// super-resolving decoder (which consumes the same unpacked latent). `height`/`width` are the
    /// requested pixel dims.
    pub fn decode(
        &self,
        latents: &Tensor,
        height: usize,
        width: usize,
        pid: Option<&PidDecoder>,
    ) -> Result<Image> {
        self.decode_with_memory(None, latents, height, width, pid, &CancelFlag::default())
    }

    pub fn decode_with_memory(
        &self,
        heavy: Option<&FluxRefHeavy>,
        latents: &Tensor,
        height: usize,
        width: usize,
        pid: Option<&PidDecoder>,
        cancel: &CancelFlag,
    ) -> Result<Image> {
        candle_gen::check_cancel(cancel)?;
        self.validate_native_vae_request(pid.is_some(), cancel)?;
        if let Some(heavy) = heavy {
            return self.pipeline.decode_ref_residency(
                &heavy.heavy,
                latents,
                height,
                width,
                pid,
                cancel,
                self.memory,
            );
        }
        let components = self.components.as_ref().ok_or_else(|| {
            candle_gen::CandleError::Msg(
                "FLUX staged reference decode requires its heavy phase".into(),
            )
        })?;
        self.pipeline
            .decode_ref(components, latents, height, width, pid)
    }
}
