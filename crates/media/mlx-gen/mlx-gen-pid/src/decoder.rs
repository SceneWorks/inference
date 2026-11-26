//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam). It carries this generation's
//! caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged trait method
//! (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes the
//! **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

use mlx_rs::{Array, Dtype};

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::gen_core::{LatentSpace, RenderRun};
use mlx_gen::{CancelFlag, Error, Result};

use crate::lq::PidNet;
use crate::plan::{PidDecodePlan, PidTilingPlan};
use crate::sampler::Sampler;

/// A PiD decoder bound to one generation's caption + σ + scale.
pub struct PidDecoder {
    net: PidNet,
    sampler: Sampler,
    /// `[1, L, txt_embed_dim]` caption embeddings for this generation (from [`crate::caption`]).
    caption_embs: Array,
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
    /// Cooperative cancellation for the ~100 s 4-step decode (F-006). Bound at decoder-mint time
    /// from `req.cancel` so [`LatentDecoder::decode`] — whose trait signature carries no flag — can
    /// still honor a cancel per sampler step. `None` ⇒ uncancellable (direct struct-API construction).
    cancel: Option<CancelFlag>,
    /// Spatial-tiling policy `(tile_edge_px, overlap_px)` for large outputs (sc-10087). Bound at
    /// mint time from the budget/watchdog plan ([`crate::budget::plan_tile_edge`]); `decode` tiles when
    /// the output exceeds `tile_edge` on either axis, else takes the exact whole-image path. `None` ⇒
    /// always whole-image (small outputs, or a direct struct-API caller that hasn't opted in).
    tile: Option<(i32, i32)>,
    /// Validated pre-decode plan used when creating standalone S10 (`RenderStage::Decode`) runs.
    plan: PidDecodePlan,
}

impl PidDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        net: PidNet,
        sampler: Sampler,
        caption_embs: Array,
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

    /// Evaluate this generation's caption embeddings now. MLX builds the Gemma caption forward
    /// lazily when the decoder is minted; a provider calls this inside its Conditioning phase so
    /// that text-encoder work is attributed to Conditioning instead of running inside the first
    /// decode. Evaluating earlier computes the same values and copies nothing.
    pub fn materialize_caption(&self) -> Result<()> {
        self.caption_embs.eval()?;
        Ok(())
    }

    /// Bind a cooperative cancellation handle (F-006). Callers that go through
    /// [`crate::resolve_pid_decoder`] get this wired from `req.cancel` automatically; direct
    /// struct-API callers (e.g. InstantID) can opt in with their request's flag.
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
    /// `tile_edge` on either axis, and takes the exact whole-image path otherwise. Callers through
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

    /// The output pixel resolution for a latent grid `[.., .., zH, zW]`.
    pub fn target_hw(&self, latents: &Array) -> (i32, i32) {
        let sh = latents.shape();
        let f = self.vae_compression * self.scale;
        (sh[2] * f, sh[3] * f)
    }

    /// Validate the `[B, z, zH, zW]` LQ-latent contract (F-100) and prepare the shared decode inputs:
    /// the bf16-cast latent, batch `b`, output `(th, tw)`, and the per-sample degrade σ.
    fn prep(&self, latents: &Array) -> Result<(Array, i32, i32, i32, Array)> {
        let sh = latents.shape();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "pid decode: expected a rank-4 [B, z, zH, zW] LQ latent, got shape {sh:?}"
            )));
        }
        let expected_z = self.net.lq_latent_channels();
        if sh[1] != expected_z {
            return Err(Error::Msg(format!(
                "pid decode: LQ latent has {} channels, expected {expected_z}",
                sh[1]
            )));
        }
        let b = sh[0];
        let (th, tw) = self.target_hw(latents);
        mlx_gen::image_execution::scalar("pid_degrade_sigma", self.sigma);
        mlx_gen::image_execution::policy_debug("pid_output_geometry", &(b, th, tw));
        mlx_gen::image_execution::policy_debug("pid_scale", &(self.scale, self.vae_compression));
        // PiD runs the released bf16 inference dtype, and the LQ-adapter convs require their input in
        // that dtype. An engine may hand us an f32 sampler latent (Qwen/Krea keep latents f32 through
        // the denoise loop), so cast here — a no-op when the latent is already bf16. Matches the
        // validated `from_clean` path (sc-7843), which cast the VAE latent to bf16 before decode.
        let latents = latents.as_dtype(Dtype::Bfloat16)?;
        let sigma = Array::from_slice(&vec![self.sigma; b as usize], &[b]);
        Ok((latents, b, th, tw, sigma))
    }

    /// Repro capture (sc-10087): when `PID_CAPTURE_LATENT` names a `.safetensors` path, dump the exact
    /// LQ latent + this generation's caption embeddings (plus σ/scale/vae_compression/seed as metadata)
    /// the *first* time `decode` runs, so a real production latent can be replayed offline (e.g. the
    /// tiled-vs-whole A/B) without re-running the full engine. Best-effort: a failure logs but never
    /// breaks the decode. No-op (one env read) when unset.
    fn maybe_capture(&self, latents: &Array) {
        let Ok(path) = std::env::var("PID_CAPTURE_LATENT") else {
            return;
        };
        if std::path::Path::new(&path).exists() {
            return; // capture once
        }
        // Best-effort (F-154): the dtype casts can fail (an exotic input dtype), so fold them into the
        // same fallible log path as the save — a capture failure must never `unwrap()`-panic the
        // production decode this env var is merely observing.
        match self.try_capture(latents, &path) {
            Ok(()) => eprintln!("[pid] captured LQ latent + caption to {path}"),
            Err(e) => eprintln!("[pid] PID_CAPTURE_LATENT capture failed: {e}"),
        }
    }

    /// The fallible body of [`Self::maybe_capture`] — casts + save, all through `?` so any failure is
    /// surfaced to the best-effort log path rather than panicking mid-decode.
    fn try_capture(&self, latents: &Array, path: &str) -> Result<()> {
        let meta = std::collections::HashMap::from([
            ("sigma".to_string(), self.sigma.to_string()),
            ("scale".to_string(), self.scale.to_string()),
            (
                "vae_compression".to_string(),
                self.vae_compression.to_string(),
            ),
            ("seed".to_string(), self.seed.to_string()),
        ]);
        let arrays = [
            ("latent", latents.as_dtype(Dtype::Float32)?),
            ("caption", self.caption_embs.as_dtype(Dtype::Float32)?),
        ];
        Array::save_safetensors(arrays.iter().map(|(k, v)| (*k, v)), &meta, path)?;
        Ok(())
    }

    /// Spatially-tiled decode (sc-10087): same result geometry + seeded noise as [`LatentDecoder::decode`],
    /// but the per-step velocity forward runs on overlapping `tile`-px pixel windows (feather-blended) so
    /// the whole-image `PidNet::forward` peak / long Metal command buffer never materializes. `tile` /
    /// `overlap` are output-pixel units.
    pub fn decode_tiled(&self, latents: &Array, tile: i32, overlap: i32) -> Result<Array> {
        self.decode_in_run_impl(latents, Some((tile, overlap)), None, None)
    }

    fn decode_in_run_impl(
        &self,
        latents: &Array,
        tile_override: Option<(i32, i32)>,
        run: Option<&RenderRun<'_, '_>>,
        cancel_override: Option<&CancelFlag>,
    ) -> Result<Array> {
        let cancel = cancel_override.or(self.cancel.as_ref());
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        // Capture the LQ latent symmetrically on tiled and whole-image paths (F-154).
        self.maybe_capture(latents);
        let (latents, b, th, tw, sigma) = self.prep(latents)?;
        let tiling = match tile_override {
            Some((tile, overlap)) => {
                if tile <= 0 || overlap < 0 || overlap >= tile {
                    return Err(Error::Msg(format!(
                        "pid decode: invalid tile ({tile}) / overlap ({overlap}) geometry"
                    )));
                }
                Some((tile, overlap))
            }
            None => self
                .tile
                .and_then(|(edge, overlap)| (th > edge || tw > edge).then_some((edge, overlap))),
        };
        let (noise, eps) = self.sampler.draw_noise_eps(b, th, tw, self.seed)?;
        let out = match run {
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
            )?,
            None => {
                let s10_plan = self
                    .plan
                    .render_plan(tw.max(1) as u32, th.max(1) as u32, b.max(1) as u32)
                    .map_err(Error::from)?;
                let fallback_cancel = CancelFlag::new();
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
                local_run.finish_output::<_, Error>(decoded)?
            }
        };
        self.sampler.observe_execution(self.seed);
        Ok(out)
    }
}

impl LatentDecoder for PidDecoder {
    fn input_latent_space(&self) -> Option<&LatentSpace> {
        self.input_latent_space.as_ref()
    }

    /// `latents`: the normalized VAE latent `[B, C, zH, zW]`. Returns super-resolved pixels
    /// `[B, 3, zH·vae_compression·scale, zW·vae_compression·scale]` in `[-1, 1]`.
    fn decode(&self, latents: &Array) -> Result<Array> {
        self.decode_in_run_impl(latents, None, None, None)
    }

    fn decode_in_run(
        &self,
        latents: &Array,
        run: Option<&RenderRun<'_, '_>>,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.decode_in_run_impl(latents, None, run, cancel)
    }
}
