//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam). It carries this generation's
//! caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged trait method
//! (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes the
//! **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use mlx_rs::{Array, Dtype};

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::driver::DriverRunState;
use mlx_gen::gen_core::{LatentSpace, RenderRun};
use mlx_gen::{CancelFlag, Error, Result};

use crate::lq::PidNet;
use crate::plan::{PidDecodePlan, PidTilingPlan};
use crate::sampler::Sampler;

#[cfg(test)]
type MockPredictFn = Box<dyn Fn(&Array, &Array) -> Result<Array> + Send + Sync>;

#[cfg(not(test))]
enum DecoderBackend {
    Net(PidNet),
}

#[cfg(test)]
enum DecoderBackend {
    Net(Box<PidNet>),
    Mock {
        lq_channels: i32,
        predict: MockPredictFn,
        draw_calls: AtomicUsize,
        capture_calls: AtomicUsize,
    },
}

#[cfg(not(test))]
impl DecoderBackend {
    fn from_net(net: PidNet) -> Self {
        Self::Net(net)
    }

    fn lq_latent_channels(&self) -> i32 {
        match self {
            Self::Net(net) => net.lq_latent_channels(),
        }
    }

    fn record_draw(&self) {}

    fn record_capture(&self) {}

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity(
        &self,
        tiling: Option<(i32, i32)>,
        x: &Array,
        t_scaled: &Array,
        caption_embs: &Array,
        latents: &Array,
        sigma: &Array,
        feather_cache: &crate::tiling::FeatherCache,
    ) -> Result<Array> {
        match (self, tiling) {
            (Self::Net(net), Some((tile, overlap))) => crate::tiling::forward_tiled(
                net,
                x,
                t_scaled,
                caption_embs,
                latents,
                sigma,
                tile,
                overlap,
                feather_cache,
            ),
            (Self::Net(net), None) => net.forward(x, t_scaled, caption_embs, latents, sigma),
        }
    }
}

#[cfg(test)]
impl DecoderBackend {
    fn from_net(net: PidNet) -> Self {
        Self::Net(Box::new(net))
    }

    fn lq_latent_channels(&self) -> i32 {
        match self {
            Self::Net(net) => net.lq_latent_channels(),
            Self::Mock { lq_channels, .. } => *lq_channels,
        }
    }

    fn record_draw(&self) {
        if let Self::Mock { draw_calls, .. } = self {
            draw_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn record_capture(&self) {
        if let Self::Mock { capture_calls, .. } = self {
            capture_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity(
        &self,
        tiling: Option<(i32, i32)>,
        x: &Array,
        t_scaled: &Array,
        caption_embs: &Array,
        latents: &Array,
        sigma: &Array,
        feather_cache: &crate::tiling::FeatherCache,
    ) -> Result<Array> {
        match (self, tiling) {
            (Self::Net(net), Some((tile, overlap))) => crate::tiling::forward_tiled(
                net,
                x,
                t_scaled,
                caption_embs,
                latents,
                sigma,
                tile,
                overlap,
                feather_cache,
            ),
            (Self::Net(net), None) => net.forward(x, t_scaled, caption_embs, latents, sigma),
            (Self::Mock { predict, .. }, _) => predict(x, t_scaled),
        }
    }
}

/// A PiD decoder bound to one generation's caption + σ + scale.
pub struct PidDecoder {
    backend: DecoderBackend,
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
            backend: DecoderBackend::from_net(net),
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

    /// Validate the `[B, z, zH, zW]` LQ-latent contract (F-100) on host shapes without allocating or
    /// casting device buffers before `RenderRun` enters `MemoryPhase::Decode`.
    fn validate_latents_host(&self, latents: &Array) -> Result<(i32, i32, i32)> {
        let sh = latents.shape();
        if sh.len() != 4 {
            return Err(Error::Msg(format!(
                "pid decode: expected a rank-4 [B, z, zH, zW] LQ latent, got shape {sh:?}"
            )));
        }
        let expected_z = self.backend.lq_latent_channels();
        if sh[1] != expected_z {
            return Err(Error::Msg(format!(
                "pid decode: LQ latent has {} channels, expected {expected_z}",
                sh[1]
            )));
        }
        let (th, tw) = self.target_hw(latents);
        Ok((sh[0], th, tw))
    }

    /// Cast and upload decode-phase inputs and record per-decode policies inside `MemoryPhase::Decode`.
    fn prepare_device_inputs(
        &self,
        latents: &Array,
        b: i32,
        th: i32,
        tw: i32,
    ) -> Result<(Array, Array)> {
        mlx_gen::image_execution::scalar("pid_degrade_sigma", self.sigma);
        mlx_gen::image_execution::policy_debug("pid_output_geometry", &(b, th, tw));
        mlx_gen::image_execution::policy_debug("pid_scale", &(self.scale, self.vae_compression));
        // PiD runs the released bf16 inference dtype, and the LQ-adapter convs require their input in
        // that dtype. An engine may hand us an f32 sampler latent (Qwen/Krea keep latents f32 through
        // the denoise loop), so cast here — a no-op when the latent is already bf16. Matches the
        // validated `from_clean` path (sc-7843), which cast the VAE latent to bf16 before decode.
        let latents = latents.as_dtype(Dtype::Bfloat16)?;
        let sigma = Array::from_slice(&vec![self.sigma; b as usize], &[b]);
        Ok((latents, sigma))
    }

    /// Repro capture (sc-10087): when `PID_CAPTURE_LATENT` names a `.safetensors` path, dump the exact
    /// LQ latent + this generation's caption embeddings (plus σ/scale/vae_compression/seed as metadata)
    /// the *first* time `decode` runs, so a real production latent can be replayed offline (e.g. the
    /// tiled-vs-whole A/B) without re-running the full engine. Best-effort: a failure logs but never
    /// breaks the decode. No-op (one env read) when unset.
    fn maybe_capture(&self, latents: &Array) {
        self.backend.record_capture();
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
        if run.is_some_and(|r| r.cancel().is_cancelled()) {
            return Err(Error::Canceled);
        }
        let (b, th, tw) = self.validate_latents_host(latents)?;
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

        let execute_in_scope =
            |driver_state: &mut DriverRunState, run_cancel: &CancelFlag| -> Result<Array> {
                if cancel.is_some_and(CancelFlag::is_cancelled) || run_cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                // Defer capture, device casts/uploads, and noise/eps RNG draws inside
                // `MemoryPhase::Decode` so a refused/pre-canceled RenderRun allocates nothing.
                self.maybe_capture(latents);
                let (latents, sigma) = self.prepare_device_inputs(latents, b, th, tw)?;
                self.backend.record_draw();
                let (noise, eps) = self.sampler.draw_noise_eps(b, th, tw, self.seed)?;
                let out = self.sampler.drive_with_driver_state(
                    &noise,
                    eps,
                    self.seed,
                    cancel,
                    Some(run_cancel),
                    driver_state,
                    |x: &Array, t_scaled: &Array| {
                        self.backend.predict_velocity(
                            tiling,
                            x,
                            t_scaled,
                            &self.caption_embs,
                            &latents,
                            &sigma,
                            &self.sampler.feather_cache,
                        )
                    },
                )?;
                if tiling.is_none() {
                    mlx_gen::image_execution::policy_debug("pid_forward_tiled", &false);
                }
                Ok(out)
            };

        let support = crate::sampler::pid_driver_support();
        let expected_steps = self.sampler.steps();
        let out = match run {
            Some(render_run) => render_run.decode_phase_with_driver_with(
                &support,
                expected_steps,
                |driver_state| execute_in_scope(driver_state, render_run.cancel()),
            )?,
            None => {
                let s10_plan = self
                    .plan
                    .render_plan(tw.max(1) as u32, th.max(1) as u32, b.max(1) as u32)
                    .map_err(Error::from)?;
                let fallback_cancel = CancelFlag::new();
                let run_cancel = cancel.unwrap_or(&fallback_cancel);
                let local_run = RenderRun::new(&s10_plan, run_cancel, None);
                let decoded = local_run.decode_phase_with_driver_with(
                    &support,
                    expected_steps,
                    |driver_state| execute_in_scope(driver_state, local_run.cancel()),
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

#[cfg(test)]
impl PidDecoder {
    pub(crate) fn new_mock(
        lq_channels: i32,
        sampler: Sampler,
        sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
        predict: impl Fn(&Array, &Array) -> Result<Array> + Send + Sync + 'static,
    ) -> Self {
        let plan =
            PidDecodePlan::from_decoder_parts(&sampler, sigma, scale, vae_compression, seed, None);
        Self {
            backend: DecoderBackend::Mock {
                lq_channels,
                predict: Box::new(predict),
                draw_calls: AtomicUsize::new(0),
                capture_calls: AtomicUsize::new(0),
            },
            sampler,
            caption_embs: Array::from_slice(&[0.0f32], &[1, 1, 1]),
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

    pub(crate) fn test_draw_calls(&self) -> usize {
        match &self.backend {
            DecoderBackend::Mock { draw_calls, .. } => draw_calls.load(Ordering::SeqCst),
            DecoderBackend::Net(_) => 0,
        }
    }

    pub(crate) fn test_capture_calls(&self) -> usize {
        match &self.backend {
            DecoderBackend::Mock { capture_calls, .. } => capture_calls.load(Ordering::SeqCst),
            DecoderBackend::Net(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SamplerConfig;
    use mlx_gen::gen_core::{
        self, render::PackedImagePass, Capabilities, GenerationRequest, MemoryGeometry,
        MemoryPhase, MemoryRequestScope, MemoryRunOutcome, Modality, ModelDescriptor, RenderPlan,
    };
    use std::sync::{Arc, Mutex};

    struct SpyScope {
        events: Mutex<Vec<&'static str>>,
        fail_enter_decode: bool,
    }

    impl SpyScope {
        fn new(fail_enter_decode: bool) -> Self {
            Self {
                events: Mutex::new(Vec::new()),
                fail_enter_decode,
            }
        }
    }

    impl MemoryRequestScope for SpyScope {
        fn configure_request(&mut self, _request: &mut GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
            let label = match phase {
                MemoryPhase::Conditioning => "enter_Conditioning",
                MemoryPhase::Denoise => "enter_Denoise",
                MemoryPhase::Decode => "enter_Decode",
            };
            self.events.lock().unwrap().push(label);
            if self.fail_enter_decode && phase == MemoryPhase::Decode {
                return Err(gen_core::Error::Unsupported("decode scope refused".into()));
            }
            Ok(())
        }
        fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
            let label = match phase {
                MemoryPhase::Conditioning => "leave_Conditioning",
                MemoryPhase::Denoise => "leave_Denoise",
                MemoryPhase::Decode => "leave_Decode",
            };
            self.events.lock().unwrap().push(label);
            Ok(())
        }
        fn configure_decode(
            &mut self,
            _tile_edge: u32,
            _overlap: u32,
            _geometry: MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(
            &mut self,
            _first_block: u32,
            _block_count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
            Ok(())
        }
    }

    fn zero_latents() -> Array {
        Array::zeros::<f32>(&[1, 16, 2, 2]).unwrap()
    }

    fn test_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "pid",
            family: "pid",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        }
    }

    #[test]
    fn decode_in_run_honors_render_run_cancel_mid_loop_without_decoder_cancel() {
        let sampler = Sampler::new(&SamplerConfig::distill_4step());
        let run_cancel = CancelFlag::new();
        let steps_seen = Arc::new(AtomicUsize::new(0));
        let steps_cb = Arc::clone(&steps_seen);
        let cancel_cb = run_cancel.clone();

        // PidDecoder has `self.cancel == None` and receives `cancel_override: None`.
        let decoder = PidDecoder::new_mock(16, sampler, 0.0, 1, 1, 7, move |x, _t| {
            steps_cb.fetch_add(1, Ordering::SeqCst);
            cancel_cb.cancel();
            Ok(Array::zeros::<f32>(x.shape()).unwrap())
        });
        assert!(decoder.cancel.is_none());

        let s10_plan = decoder.plan().render_plan(2, 2, 1).unwrap();
        let mut scope = SpyScope::new(false);
        let render_run = RenderRun::new(&s10_plan, &run_cancel, Some(&mut scope));
        let err = match decoder.decode_in_run(&zero_latents(), Some(&render_run), None) {
            Ok(_) => panic!("expected cancellation mid-decode"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::Canceled), "got {err:?}");
        assert_eq!(
            steps_seen.load(Ordering::SeqCst),
            1,
            "tripping render_run.cancel() on step 0 must abort before step 1"
        );
        assert_eq!(decoder.test_draw_calls(), 1);
        assert_eq!(decoder.test_capture_calls(), 1);
    }

    #[test]
    fn decode_in_run_defers_capture_and_noise_draws_until_render_run_accepts_decode_scope() {
        let sampler = Sampler::new(&SamplerConfig::distill_4step());
        let steps_seen = Arc::new(AtomicUsize::new(0));
        let steps_cb = Arc::clone(&steps_seen);
        let decoder = PidDecoder::new_mock(16, sampler, 0.0, 1, 1, 9, move |x, _t| {
            steps_cb.fetch_add(1, Ordering::SeqCst);
            Ok(Array::zeros::<f32>(x.shape()).unwrap())
        });

        let desc = test_descriptor();
        let req = GenerationRequest {
            width: 8,
            height: 8,
            count: 1,
            steps: Some(1),
            ..Default::default()
        };
        let packed_plan = RenderPlan::packed_image_trajectory(
            &desc,
            &req,
            PackedImagePass {
                sampler: "euler",
                guidance: 0.0,
                sigmas: vec![1.0, 0.0],
                executed_steps: 1,
                capture_sigma: 0.0,
                pixel_stride: 1,
                token_width: 4,
            },
        )
        .unwrap();

        // 1. Pre-canceled RenderRun with decoder.cancel == None skips capture & noise draw.
        let s10_plan = decoder.plan().render_plan(2, 2, 1).unwrap();
        let canceled = CancelFlag::new();
        canceled.cancel();
        let canceled_run = RenderRun::new(&s10_plan, &canceled, None);
        assert!(matches!(
            decoder.decode_in_run(&zero_latents(), Some(&canceled_run), None),
            Err(Error::Canceled)
        ));

        // 2. Refused state (BeforeEncode) skips capture, draw_noise_eps, and predict body.
        let untripped = CancelFlag::new();
        let before_encode_run = RenderRun::new(&packed_plan, &untripped, None);
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&before_encode_run), None)
            .is_err());

        // 3. Refused active denoise phase skips capture, draw_noise_eps, and predict body.
        let active_denoise_run = RenderRun::new(&packed_plan, &untripped, None);
        active_denoise_run.encode(|| Ok::<_, Error>(())).unwrap();
        active_denoise_run.begin_rendering().unwrap();
        active_denoise_run.begin_denoise_phase(0).unwrap();
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&active_denoise_run), None)
            .is_err());

        // 4. Refused step-count mismatch (S10 compiled for 2 steps vs sampler 4) skips draw & body.
        let bad_steps_plan =
            RenderPlan::decode_stage(&desc, 2, 2, 1, 9, "pid_sde", vec![0.999, 0.5, 0.0], 2)
                .unwrap();
        let bad_steps_run = RenderRun::new(&bad_steps_plan, &untripped, None);
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&bad_steps_run), None)
            .is_err());

        // 5. Refused MemoryRequestScope enter(Decode) skips draw & body.
        let mut refusing_scope = SpyScope::new(true);
        let refused_scope_run = RenderRun::new(&s10_plan, &untripped, Some(&mut refusing_scope));
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&refused_scope_run), None)
            .is_err());

        assert_eq!(decoder.test_draw_calls(), 0);
        assert_eq!(decoder.test_capture_calls(), 0);
        assert_eq!(steps_seen.load(Ordering::SeqCst), 0);

        // 6. Valid S10 RenderRun enters Decode scope, runs 4 steps, leaves Decode scope.
        let mut good_scope = SpyScope::new(false);
        let good_run = RenderRun::new(&s10_plan, &untripped, Some(&mut good_scope));
        decoder
            .decode_in_run(&zero_latents(), Some(&good_run), None)
            .unwrap();
        assert_eq!(decoder.test_draw_calls(), 1);
        assert_eq!(decoder.test_capture_calls(), 1);
        assert_eq!(steps_seen.load(Ordering::SeqCst), 4);
        assert_eq!(
            *good_scope.events.lock().unwrap(),
            vec!["enter_Decode", "leave_Decode"]
        );
    }
}
