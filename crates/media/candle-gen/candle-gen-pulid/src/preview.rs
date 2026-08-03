//! PuLID-FLUX's per-step latent preview seam (epic 16948, sc-16956) — a **re-export**, deliberately.
//!
//! PuLID owns no VAE, no fit and no projector. It composes `candle-gen-flux`'s own FLUX.1-dev backbone
//! ([`crate::pulid_flux::PulidFlux`] holds a [`candle_gen_flux::FluxRefBackbone`]), so the latent it
//! integrates, the `flux::sampling::unpack` that recovers it and the `AutoencoderKL` that decodes it are
//! **the same code and the same weight files** the registered `flux1_dev` route uses — the tier-detecting
//! load reads `SceneWorks/flux1-dev-mlx`'s `q4`/`q8`/`bf16` subdirs or a dense BFL snapshot's
//! `ae.safetensors`, exactly as the base generator does. Its reuse of the epic-16624 16-channel fit is
//! therefore not an analogy between two similar spaces; it is the same space, reached through the same
//! loader. The container-level provenance is in [`candle_gen_flux::preview`]'s module docs and re-derived
//! per snapshot by `candle-gen-flux/tests/preview_real_weights.rs`.
//!
//! ## PuLID registers no descriptor, so there is no id to advertise
//!
//! Epic 16948's scoping expected a bespoke candle provider **id** here. There is none:
//! [`crate`]'s docs state it, `candle-gen-catalog`'s `BESPOKE_UTILITY_CRATES` lists `pulid`, and
//! `temporal_and_super_resolution_routes_stay_outside_preview_advertising` asserts by exact id that no
//! registered descriptor is ever named `pulid` or `pulid_flux`. PuLID is a plain struct the worker drives
//! by name, like InstantID and the IP-Adapters, so its preview arrives through a `preview` field on
//! [`crate::PulidFluxRequest`] rather than through `GenerationRequest`, and its route inventory is pinned
//! in the catalog's bespoke-crate table rather than in `PREVIEW_ROUTE_IDS`. Inventing a registration to
//! give it an id would be the very thing that guard exists to prevent.
//!
//! ## The identity path cannot perturb the previewed latent
//!
//! This story's PuLID-specific criterion is that the id embedding must not disturb what is previewed.
//! It is closed **structurally**, not by a check: `compute_id_embedding` runs once before the denoise and
//! its 32-token result is captured by a [`crate::ca::PulidCa`] injector that
//! [`candle_gen_flux::IpFlux::forward_injected`] applies **inside** the predict closure, between DiT
//! blocks, on the image stream. The tensor [`candle_gen::run_flow_sampler`] integrates — and therefore
//! the only tensor the hook is ever handed — is the `[1, S, 64]` image latent alone. The identity tokens
//! change the velocity that latent moves along; they never join it.
//! `candle-gen-flux`'s `injected_conditioning_never_reaches_the_previewed_latent` drives exactly that
//! shape through the real sampler, with a conditioning stream concatenated on the sequence axis and
//! sliced back out, and asserts the hook only ever sees the target sequence length.

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

    /// PuLID reuses the SIXTEEN-channel FLUX.1 fit, and its own latent-channel constant is that number.
    /// Pinned here as well as in the donor crate so a re-point of this `pub use` to another family
    /// fails in PuLID's own suite.
    #[test]
    fn the_reused_fit_is_the_sixteen_channel_flux1_one() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(PACKED_LATENT_CHANNELS, 64);
        assert_eq!(
            PREVIEW_LATENT_CHANNELS,
            crate::pulid_flux::LATENT_CHANNELS,
            "the preview fit and the noise PuLID seeds must describe the same latent"
        );
    }

    /// The preview's packed→native recovery agrees with PuLID's own noise geometry — the
    /// `⌈·/16⌉·2` latent `generate` builds — at PuLID's shipped sizes. A hook built at a different grid
    /// would produce a frame at the wrong resolution, which the shared emitter would swallow silently.
    #[test]
    fn the_preview_grid_matches_pulids_own_latent_geometry() {
        for (w, h) in [(1024u32, 1024u32), (768, 1024), (512, 512)] {
            let (rows, cols) = token_grid(w, h);
            let (lat_h, lat_w) = ((h as usize).div_ceil(16) * 2, (w as usize).div_ceil(16) * 2);
            assert_eq!((rows * 2, cols * 2), (lat_h, lat_w), "{w}x{h}");

            let tokens = Tensor::zeros(
                (1, rows * cols, PACKED_LATENT_CHANNELS),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let frame = project_packed_tokens(&tokens, w, h).unwrap();
            assert_eq!((frame.width, frame.height), (lat_w as u32, lat_h as u32));
        }
    }

    /// The one sampler site is live end to end over PuLID's own dev time-shifted schedule: one numbered
    /// frame per outer step, and an inert sink leaves the latent byte-identical. Weights-free — the DiT
    /// is a stand-in — so the row measures the wiring and nothing else.
    #[test]
    fn the_pulid_schedule_emits_one_frame_per_step_and_stays_inert_when_unhooked() {
        let (width, height) = (64u32, 64u32);
        let (rows, cols) = token_grid(width, height);
        let seq = rows * cols;
        let start =
            Tensor::rand(-1f32, 1f32, (1, seq, PACKED_LATENT_CHANNELS), &Device::Cpu).unwrap();

        // PuLID is always FLUX.1-dev: the verbatim time-shifted `get_schedule`, re-strided through the
        // shared resolver exactly as `PulidFlux::denoise` does.
        let steps = 6usize;
        let native: Vec<f32> = candle_transformers::models::flux::sampling::get_schedule(
            steps,
            Some((seq, candle_gen_flux::BASE_SHIFT, candle_gen_flux::MAX_SHIFT)),
        )
        .iter()
        .map(|&t| t as f32)
        .collect();
        let sigmas = candle_gen::resolve_flow_schedule(
            None,
            candle_gen_flux::flow_mu(candle_gen_flux::Variant::Dev, seq),
            steps,
            &native,
        );
        assert_eq!(sigmas.len(), steps + 1);

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
            (1..=steps as u32)
                .map(|n| (n, steps as u32))
                .collect::<Vec<_>>()
        );

        let inert = PreviewSink::default();
        let quiet = hook(&inert, width, height);
        assert!(!quiet.is_active());
        assert_eq!(run(Some(&quiet)), bare);
    }

    /// The crate ships exactly one denoise lane, and it passes a hook. Pinned against PuLID's own
    /// source because a `BESPOKE_UTILITY_CRATES` member has no descriptor whose `supports_preview`
    /// could otherwise contradict it — the catalog's bespoke-crate inventory is the other half.
    #[test]
    fn the_single_pulid_render_route_passes_a_preview_hook() {
        let source = include_str!("pulid_flux.rs");
        let code: String = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            code.matches("run_flow_sampler(").count(),
            1,
            "pulid_flux.rs must drive exactly one sampler site"
        );
        assert!(
            code.contains("Some(preview),"),
            "the one sampler site must pass a preview hook, not `None`"
        );
        assert!(
            !code.contains("            None,\n            |img, t|"),
            "no sampler site may pass `None` immediately before the predict closure"
        );
    }
}
