//! Chroma's per-step latent preview seam (epic 16948, sc-16956) — a **re-export**, deliberately.
//!
//! Chroma owns no fit and no projector. It denoises the same packed `[1, S, 64]` token sequence FLUX.1
//! does, recovers it with the same [`candle_transformers::models::flux::sampling::unpack`], and decodes
//! it through a VAE whose learned tensors are the very bytes epic 16624's 16-channel fit was measured
//! over: every shipped Chroma re-host — HD `9d99afe1…`, Base `e7330dda…`, Flash `6a9cb617…` — ships a
//! `vae/diffusion_pytorch_model.safetensors` **byte-identical** to `black-forest-labs/FLUX.1-dev`'s
//! (SHA-256 `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`, 167,666,902 bytes, 244
//! tensors), with the identical `scaling_factor: 0.3611` / `shift_factor: 0.1159` /
//! `latent_channels: 16` config. So the coefficients that describe FLUX.1's latent space describe
//! Chroma's, and reusing them through [`candle_gen_flux::preview`] is what keeps a second copy of them
//! from existing — the same relationship `mlx-gen-chroma` has to `mlx-gen-flux`, whose `load_vae` it
//! calls outright.
//!
//! The reuse is re-derived per snapshot by `tests/preview_real_weights.rs`; the full record is
//! `docs/migration/evidence/sc-16956-flux1-candle-preview.md`.
//!
//! ## The one thing Chroma does differently
//!
//! Chroma is the family member that runs **true CFG** — `pred = neg + g·(pos − neg)` — where FLUX.1 dev
//! is guidance-distilled and schnell has no guidance at all. That blend happens **inside**
//! `crate::pipeline`'s predict closure (a private module, hence no intra-doc link) and returns a
//! single velocity in the conditional space, so the
//! tensor [`candle_gen::run_flow_sampler`] integrates is never a fused `[2, …]` batch and the preview
//! can never project the unconditional half. `candle-gen-flux`'s
//! `cfg_never_exposes_the_unconditional_half_to_the_preview` drives exactly this closure shape through
//! the real sampler.

pub use candle_gen_flux::preview::{
    hook, project_packed_tokens, project_raw_latents, token_grid, PACKED_LATENT_CHANNELS,
    PREVIEW_LATENT_CHANNELS,
};

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::gen_core::{PreviewFrame, PreviewSink};

    use super::*;

    /// Chroma reuses the SIXTEEN-channel FLUX.1 fit, not FLUX.2's thirty-two-channel one. Pinned here
    /// as well as in the donor crate so a re-point of this `pub use` to the wrong family fails in
    /// Chroma's own suite.
    #[test]
    fn the_reused_fit_is_the_sixteen_channel_flux1_one() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(PACKED_LATENT_CHANNELS, 64);
        assert_eq!(PREVIEW_LATENT_CHANNELS, crate::pipeline::LATENT_CHANNELS);
    }

    /// The preview's packed→native recovery and Chroma's own decode agree on geometry, at Chroma's own
    /// sizes. Both call `flux::sampling::unpack`, so this pins the pair rather than one of them: a
    /// preview built at the wrong grid would be a frame at the wrong resolution, which is precisely
    /// what the shared emitter would swallow silently if the check were not here.
    #[test]
    fn the_preview_grid_matches_chromas_own_latent_geometry() {
        for (w, h) in [(1024u32, 1024u32), (768, 512), (256, 256)] {
            let (rows, cols) = token_grid(w, h);
            // `Pipeline::initial_packed_noise` builds a [1, 16, lat_h, lat_w] latent and packs it 2x2.
            let (lat_h, lat_w) = ((h as usize).div_ceil(16) * 2, (w as usize).div_ceil(16) * 2);
            assert_eq!((rows * 2, cols * 2), (lat_h, lat_w), "{w}x{h}");

            let tokens = Tensor::zeros(
                (1, rows * cols, PACKED_LATENT_CHANNELS),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let frame = project_packed_tokens(&tokens, w, h).unwrap();
            assert_eq!(
                (frame.width, frame.height),
                (lat_w as u32, lat_h as u32),
                "{w}x{h} must preview at Chroma's own native latent resolution"
            );
        }
    }

    /// The registered route's hook is live end to end: a real Chroma σ schedule driven through the real
    /// flow sampler emits one numbered frame per step, and an inert sink emits none while leaving the
    /// latent untouched. Weights-free — the DiT is a zero-velocity stand-in, so the row measures the
    /// wiring and nothing else.
    #[test]
    fn a_chroma_schedule_emits_one_frame_per_step_and_stays_inert_when_unhooked() {
        let (width, height) = (64u32, 64u32);
        let (rows, cols) = token_grid(width, height);
        let start = Tensor::rand(
            -1f32,
            1f32,
            (1, rows * cols, PACKED_LATENT_CHANNELS),
            &Device::Cpu,
        )
        .unwrap();
        // Chroma HD's own schedule (static shift 3), taken from the shipped pipeline rather than
        // hand-written, so the row cannot pass against a schedule Chroma never runs. `Pipeline::load`
        // is lazy — it records paths and reads no weights — so this needs neither a snapshot nor a GPU.
        let tmp = tempfile::tempdir().unwrap();
        let pipe = crate::pipeline::Pipeline::load(
            crate::ChromaVariant::Hd,
            tmp.path(),
            &Device::Cpu,
            None,
        );
        let sigmas = pipe.sigmas(6);
        assert_eq!(sigmas.len(), 7);
        assert!(sigmas.windows(2).all(|w| w[0] > w[1]));

        let run = |preview: Option<&candle_gen::preview::PreviewHook<'_>>| {
            candle_gen::run_flow_sampler(
                None,
                candle_gen::gen_core::sampling::TimestepConvention::Sigma,
                &sigmas,
                start.clone(),
                16956,
                &candle_gen::gen_core::CancelFlag::new(),
                &mut |_: candle_gen::gen_core::Progress| {},
                preview,
                |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };

        let bare = run(None);

        let frames: Arc<Mutex<Vec<PreviewFrame>>> = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        let live = hook(&sink, width, height);
        assert_eq!(
            run(Some(&live)),
            bare,
            "a live sink must not move the latent"
        );
        assert_eq!(
            candle_gen::lock_recover(&frames)
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            (1..=6).map(|n| (n, 6)).collect::<Vec<_>>()
        );

        let inert = PreviewSink::default();
        let quiet = hook(&inert, width, height);
        assert!(!quiet.is_active());
        assert_eq!(run(Some(&quiet)), bare);
    }
}
