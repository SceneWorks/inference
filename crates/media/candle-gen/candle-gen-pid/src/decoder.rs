//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam, candle mirror). It carries this
//! generation's caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged
//! trait method (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes
//! the **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{LatentSpace, RenderRun};
use candle_gen::{CandleError, LatentDecoder, Result};

use crate::lq::PidNet;
use crate::plan::{PidDecodePlan, PidTilingPlan};
use crate::sampler::Sampler;

/// A PiD decoder bound to one generation's caption + σ + scale.
pub struct PidDecoder {
    net: PidNet,
    sampler: Sampler,
    /// `[1, L, txt_embed_dim]` caption embeddings for this generation (from [`crate::caption`]), f32.
    caption_embs: Tensor,
    /// Degrade σ fed to the LQ gate (0 for a clean-latent decode).
    sigma: f32,
    /// Spatial SR factor (4× for the released students).
    scale: i32,
    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    vae_compression: i32,
    /// Typed input contract. Direct test construction starts unknown; production engines bind the
    /// registry-derived space through [`Self::with_input_latent_space`].
    input_latent_space: Option<LatentSpace>,
    /// Per-decode RNG seed for the sampler's noise + per-step ε.
    seed: u64,
    /// Cooperative cancellation for the multi-second 4-step decode (F-006). Bound at decoder-mint time
    /// from `req.cancel`. `None` ⇒ uncancellable (direct struct-API construction).
    cancel: Option<CancelFlag>,
    /// Spatial-tiling policy `(tile_edge_px, overlap_px)` for large outputs (sc-10087). Bound at mint
    /// time from the budget plan ([`crate::budget::plan_tile_edge`]); `decode` tiles when the output
    /// exceeds `tile_edge` on either axis (which on CUDA is what avoids the sysmem-fallback silent hang),
    /// else takes the exact whole-image path. `None` ⇒ always whole-image.
    tile: Option<(i32, i32)>,
    /// Validated pre-decode plan used when creating standalone S10 (`RenderStage::Decode`) runs.
    plan: PidDecodePlan,
}

impl PidDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        net: PidNet,
        sampler: Sampler,
        caption_embs: Tensor,
        sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
    ) -> Self {
        let plan =
            PidDecodePlan::from_decoder_parts(&sampler, sigma, scale, vae_compression, seed, None);
        Self {
            net,
            sampler,
            caption_embs,
            sigma,
            scale,
            vae_compression,
            input_latent_space: None,
            seed,
            cancel: None,
            tile: None,
            plan,
        }
    }

    pub(crate) fn with_plan(mut self, plan: PidDecodePlan) -> Self {
        self.plan = plan;
        self
    }

    /// The compiled [`PidDecodePlan`] backing this decoder.
    pub fn plan(&self) -> &PidDecodePlan {
        &self.plan
    }

    /// Bind a cooperative cancellation handle (F-006). Callers that go through
    /// [`crate::resolve_pid_decoder`] get this wired from `req.cancel` automatically.
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn with_input_latent_space(mut self, latent_space: LatentSpace) -> Self {
        self.input_latent_space = Some(latent_space);
        self
    }

    /// Enable spatial tiling for large outputs (sc-10087): [`LatentDecoder::decode`] tiles the per-step
    /// velocity forward with `tile_edge`-px tiles (feather `overlap`) whenever the output exceeds
    /// `tile_edge` on either axis, else takes the exact whole-image path. Callers through
    /// [`crate::resolve_pid_decoder`] get this wired from [`crate::budget::plan_tile_edge`] automatically.
    pub fn with_tiling(mut self, tile_edge: i32, overlap: i32) -> Self {
        self.tile = Some((tile_edge, overlap));
        self.plan.tiling = PidTilingPlan {
            tile_edge_px: tile_edge,
            overlap_px: overlap,
            whole_fits: false,
            selected_by_request: self.plan.tiling.selected_by_request,
        };
        self
    }

    /// The output pixel resolution `(H, W)` for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Tensor) -> Result<(usize, usize)> {
        let (_, _, zh, zw) = latents.dims4()?;
        let f = (self.vae_compression * self.scale) as usize;
        Ok((zh * f, zw * f))
    }

    /// Spatially-tiled decode (sc-10087): same result geometry + seeded noise as [`LatentDecoder::decode`],
    /// but the per-step velocity forward runs on overlapping `tile`-px pixel windows (feather-blended), so
    /// the whole-image `PidNet::forward` VRAM peak never materializes. `tile`/`overlap` are output-pixel
    /// units. Used above the budget threshold at the decode seam, and by the A/B harness.
    pub fn decode_tiled(&self, latents: &Tensor, tile: i32, overlap: i32) -> Result<Tensor> {
        self.decode_in_run_impl(latents, Some((tile, overlap)), None, None)
    }

    fn decode_in_run_impl(
        &self,
        latents: &Tensor,
        tile_override: Option<(i32, i32)>,
        run: Option<&RenderRun<'_, '_>>,
        cancel_override: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let cancel = cancel_override.or(self.cancel.as_ref());
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(CandleError::Canceled);
        }
        candle_gen::execution::policy(
            "pid_degrade",
            format!(
                "sigma_bits={};scale={};vae_compression={}",
                self.sigma.to_bits(),
                self.scale,
                self.vae_compression
            )
            .into(),
        );
        let b = latents.dim(0)?;
        let (th, tw) = self.target_hw(latents)?;
        // The PiD net runs f32; an engine may hand an f16/bf16 sampler latent, so cast here (a no-op
        // when already f32). Matches the validated `from_clean` handoff (the normalized VAE latent).
        let latents = latents.to_dtype(DType::F32)?;
        let sigma = Tensor::from_vec(vec![self.sigma; b], (b,), latents.device())?;
        let tiling = match tile_override {
            Some((tile, overlap)) => {
                if tile <= 0 || overlap < 0 || overlap >= tile {
                    return Err(CandleError::Msg(format!(
                        "pid decode: invalid tile ({tile}) / overlap ({overlap}) geometry"
                    )));
                }
                Some((tile, overlap))
            }
            None => self.tile.and_then(|(edge, overlap)| {
                (th as i32 > edge || tw as i32 > edge).then_some((edge, overlap))
            }),
        };
        let (noise, eps) = self
            .sampler
            .draw_noise_eps(latents.device(), b, th, tw, self.seed)?;
        match run {
            Some(render_run) => self.sampler.run_in_render_run(
                &self.net,
                &noise,
                eps,
                &self.caption_embs,
                &latents,
                &sigma,
                self.seed,
                tiling,
                cancel,
                render_run,
            ),
            None => {
                let s10_plan = self
                    .plan
                    .render_plan(tw.max(1) as u32, th.max(1) as u32, b.max(1) as u32)
                    .map_err(CandleError::from)?;
                let fallback_cancel = CancelFlag::default();
                let run_cancel = cancel.unwrap_or(&fallback_cancel);
                let local_run = RenderRun::new(&s10_plan, run_cancel, None);
                let decoded = self.sampler.run_in_render_run(
                    &self.net,
                    &noise,
                    eps,
                    &self.caption_embs,
                    &latents,
                    &sigma,
                    self.seed,
                    tiling,
                    Some(run_cancel),
                    &local_run,
                )?;
                local_run.finish_output::<_, CandleError>(decoded)
            }
        }
    }
}

impl LatentDecoder for PidDecoder {
    fn input_latent_space(&self) -> Option<&LatentSpace> {
        self.input_latent_space.as_ref()
    }

    /// `latents`: the normalized VAE latent `[B, C, zH, zW]`. Returns super-resolved pixels
    /// `[B, 3, zH·vae_compression·scale, zW·vae_compression·scale]` in `[-1, 1]` (f32).
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_in_run_impl(latents, None, None, None)
    }

    fn decode_in_run(
        &self,
        latents: &Tensor,
        run: Option<&RenderRun<'_, '_>>,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        self.decode_in_run_impl(latents, None, run, cancel)
    }
}
