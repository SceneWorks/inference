//! The PiD [`LatentDecoder`] — the per-generation decoder that swaps an engine's `vae.decode(latent)`
//! for a super-resolving PiD pixel-diffusion decode (the sc-7844 seam, candle mirror). It carries this
//! generation's caption embeddings (+ degrade σ + SR scale), so `decode(latents)` stays the unchanged
//! trait method (the engine already holds the conditioning). Faithful to `from_clean.py`: PiD consumes
//! the **normalized** VAE latent directly; the output resolution is `latent_grid · vae_compression · scale`.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::driver::DriverRunState;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{LatentSpace, RenderRun};
use candle_gen::{CandleError, LatentDecoder, Result};

use crate::lq::PidNet;
use crate::plan::{PidDecodePlan, PidTilingPlan};
use crate::sampler::Sampler;

#[cfg(test)]
type MockPredictFn = Box<dyn Fn(&Tensor, &Tensor) -> Result<Tensor> + Send + Sync>;

#[cfg(not(test))]
enum DecoderBackend {
    Net(PidNet),
}

#[cfg(test)]
enum DecoderBackend {
    Net(Box<PidNet>),
    Mock {
        predict: MockPredictFn,
        draw_calls: AtomicUsize,
    },
}

#[cfg(not(test))]
impl DecoderBackend {
    fn from_net(net: PidNet) -> Self {
        Self::Net(net)
    }

    fn record_draw(&self) {}

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity(
        &self,
        tiling: Option<(i32, i32)>,
        x: &Tensor,
        t_scaled: &Tensor,
        caption_embs: &Tensor,
        latents: &Tensor,
        sigma: &Tensor,
    ) -> Result<Tensor> {
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

    fn record_draw(&self) {
        if let Self::Mock { draw_calls, .. } = self {
            draw_calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_velocity(
        &self,
        tiling: Option<(i32, i32)>,
        x: &Tensor,
        t_scaled: &Tensor,
        caption_embs: &Tensor,
        latents: &Tensor,
        sigma: &Tensor,
    ) -> Result<Tensor> {
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

    fn validate_latents_host(&self, latents: &Tensor) -> Result<(usize, usize, usize)> {
        let (b, _c, _zh, _zw) = latents.dims4()?;
        let (th, tw) = self.target_hw(latents)?;
        Ok((b, th, tw))
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
        if run.is_some_and(|r| r.cancel().is_cancelled()) {
            return Err(CandleError::Canceled);
        }
        let (b, th, tw) = self.validate_latents_host(latents)?;
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

        let execute_in_scope =
            |driver_state: &mut DriverRunState, run_cancel: &CancelFlag| -> Result<Tensor> {
                if cancel.is_some_and(CancelFlag::is_cancelled) || run_cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                // Defer policy emission, device dtype casts/uploads, and noise/eps RNG draws inside
                // `MemoryPhase::Decode` so a refused/pre-canceled RenderRun allocates nothing.
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
                // The PiD net runs f32; an engine may hand an f16/bf16 sampler latent, so cast here
                // (a no-op when already f32). Matches the validated `from_clean` handoff.
                let latents = latents.to_dtype(DType::F32)?;
                let sigma = Tensor::from_vec(vec![self.sigma; b], (b,), latents.device())?;
                self.backend.record_draw();
                let (noise, eps) =
                    self.sampler
                        .draw_noise_eps(latents.device(), b, th, tw, self.seed)?;
                match tiling {
                    Some((tile, overlap)) => candle_gen::execution::policy(
                        "pid_decode",
                        format!("tiled;edge={tile};overlap={overlap}").into(),
                    ),
                    None => candle_gen::execution::policy("pid_decode", "whole_image".into()),
                }
                self.sampler.drive_with_driver_state(
                    &noise,
                    eps,
                    self.seed,
                    cancel,
                    Some(run_cancel),
                    driver_state,
                    |x: &Tensor, t_scaled: &Tensor| {
                        self.backend.predict_velocity(
                            tiling,
                            x,
                            t_scaled,
                            &self.caption_embs,
                            &latents,
                            &sigma,
                        )
                    },
                )
            };

        let support = crate::sampler::pid_driver_support();
        let expected_steps = self.sampler.steps();
        match run {
            Some(render_run) => {
                render_run.decode_phase_with_driver_with(&support, expected_steps, |driver_state| {
                    execute_in_scope(driver_state, render_run.cancel())
                })
            }
            None => {
                let s10_plan = self
                    .plan
                    .render_plan(tw.max(1) as u32, th.max(1) as u32, b.max(1) as u32)
                    .map_err(CandleError::from)?;
                let fallback_cancel = CancelFlag::default();
                let run_cancel = cancel.unwrap_or(&fallback_cancel);
                let local_run = RenderRun::new(&s10_plan, run_cancel, None);
                let decoded = local_run.decode_phase_with_driver_with(
                    &support,
                    expected_steps,
                    |driver_state| execute_in_scope(driver_state, local_run.cancel()),
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

#[cfg(test)]
impl PidDecoder {
    pub(crate) fn new_mock(
        sampler: Sampler,
        sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
        predict: impl Fn(&Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Self {
        let plan =
            PidDecodePlan::from_decoder_parts(&sampler, sigma, scale, vae_compression, seed, None);
        let caption_embs =
            Tensor::zeros((1, 1, 1), DType::F32, &candle_gen::candle_core::Device::Cpu)
                .expect("CPU zero caption stub");
        Self {
            backend: DecoderBackend::Mock {
                predict: Box::new(predict),
                draw_calls: AtomicUsize::new(0),
            },
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

    pub(crate) fn test_draw_calls(&self) -> usize {
        match &self.backend {
            DecoderBackend::Mock { draw_calls, .. } => draw_calls.load(Ordering::SeqCst),
            DecoderBackend::Net(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SamplerConfig;
    use candle_gen::candle_core::Device;
    use candle_gen::gen_core::{
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
            candle_gen::lock_recover(&self.events).push(label);
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
            candle_gen::lock_recover(&self.events).push(label);
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

    fn zero_latents() -> Tensor {
        Tensor::zeros((1, 16, 2, 2), DType::F32, &Device::Cpu).unwrap()
    }

    fn test_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "pid",
            family: "pid",
            backend: "candle",
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
        let run_cancel = CancelFlag::default();
        let steps_seen = Arc::new(AtomicUsize::new(0));
        let steps_cb = Arc::clone(&steps_seen);
        let cancel_cb = run_cancel.clone();

        let decoder = PidDecoder::new_mock(sampler, 0.0, 1, 1, 7, move |x, _t| {
            steps_cb.fetch_add(1, Ordering::SeqCst);
            cancel_cb.cancel();
            Ok(Tensor::zeros(x.dims(), DType::F32, x.device())?)
        });
        assert!(decoder.cancel.is_none());

        let s10_plan = decoder.plan().render_plan(2, 2, 1).unwrap();
        let mut scope = SpyScope::new(false);
        let render_run = RenderRun::new(&s10_plan, &run_cancel, Some(&mut scope));
        let err = decoder
            .decode_in_run(&zero_latents(), Some(&render_run), None)
            .unwrap_err();
        assert!(matches!(err, CandleError::Canceled), "got {err:?}");
        assert_eq!(
            steps_seen.load(Ordering::SeqCst),
            1,
            "tripping render_run.cancel() on step 0 must abort before step 1"
        );
        assert_eq!(decoder.test_draw_calls(), 1);
    }

    #[test]
    fn decode_in_run_defers_uploads_and_noise_draws_until_render_run_accepts_decode_scope() {
        let sampler = Sampler::new(&SamplerConfig::distill_4step());
        let steps_seen = Arc::new(AtomicUsize::new(0));
        let steps_cb = Arc::clone(&steps_seen);
        let decoder = PidDecoder::new_mock(sampler, 0.0, 1, 1, 9, move |x, _t| {
            steps_cb.fetch_add(1, Ordering::SeqCst);
            Ok(Tensor::zeros(x.dims(), DType::F32, x.device())?)
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

        let s10_plan = decoder.plan().render_plan(2, 2, 1).unwrap();
        let canceled = CancelFlag::default();
        canceled.cancel();
        let canceled_run = RenderRun::new(&s10_plan, &canceled, None);
        assert!(matches!(
            decoder.decode_in_run(&zero_latents(), Some(&canceled_run), None),
            Err(CandleError::Canceled)
        ));

        let untripped = CancelFlag::default();
        let before_encode_run = RenderRun::new(&packed_plan, &untripped, None);
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&before_encode_run), None)
            .is_err());

        let active_denoise_run = RenderRun::new(&packed_plan, &untripped, None);
        active_denoise_run
            .encode(|| Ok::<_, CandleError>(()))
            .unwrap();
        active_denoise_run.begin_rendering().unwrap();
        active_denoise_run.begin_denoise_phase(0).unwrap();
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&active_denoise_run), None)
            .is_err());

        let bad_steps_plan =
            RenderPlan::decode_stage(&desc, 2, 2, 1, 9, "pid_sde", vec![0.999, 0.5, 0.0], 2)
                .unwrap();
        let bad_steps_run = RenderRun::new(&bad_steps_plan, &untripped, None);
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&bad_steps_run), None)
            .is_err());

        let mut refusing_scope = SpyScope::new(true);
        let refused_scope_run = RenderRun::new(&s10_plan, &untripped, Some(&mut refusing_scope));
        assert!(decoder
            .decode_in_run(&zero_latents(), Some(&refused_scope_run), None)
            .is_err());

        assert_eq!(decoder.test_draw_calls(), 0);
        assert_eq!(steps_seen.load(Ordering::SeqCst), 0);

        let mut good_scope = SpyScope::new(false);
        let good_run = RenderRun::new(&s10_plan, &untripped, Some(&mut good_scope));
        decoder
            .decode_in_run(&zero_latents(), Some(&good_run), None)
            .unwrap();
        assert_eq!(decoder.test_draw_calls(), 1);
        assert_eq!(steps_seen.load(Ordering::SeqCst), 4);
        assert_eq!(
            *candle_gen::lock_recover(&good_scope.events),
            vec!["enter_Decode", "leave_Decode"]
        );
    }
}
