//! The `minimax_h3` model descriptor.

use super::*;

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        // Both `None`, and both deliberate (sc-17137 main sync):
        //   * `encoder_contract`: like every shipped video provider (Wan, LTX, Mochi, SVD,
        //     SeedVR2), the text encoder is not advertised as substitutable.
        //   * `denoiser_output_latent_space`: the denoiser's video half is a 24-channel latent on
        //     the 17-frame clip lattice — `ceil(clip/vae_ratio_t) - token_drop` tokens per clip,
        //     decoded twice with a seam cross-fade (`crate::chunking`). No `LatentTemporalLaw`
        //     variant expresses that mapping, and per the field's contract `None` makes every
        //     external decoder fail closed rather than let a channel-count match imply
        //     compatibility. The paired ViT decoder + audio VAE are internal to this crate.
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "minimax_h3",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch exists anywhere in the checkpoint.
            supports_negative_prompt: false,
            // **One kind covers all three keyframe shapes.** `Keyframe` carries a `frame_idx`, and
            // first-only / last-only / first+last are one, one and two of them — see
            // [`keyframe_anchors`] for why last-frame-only is a payload shape rather than a mode
            // of its own.
            //
            // The three `ref2va` kinds (sc-17149) are the omni-reference surface on
            // `transformer_ref`. They are three kinds rather than one because gen-core has no
            // heterogeneous-reference variant and `conditioning` is an **ordered** `Vec`, so the
            // request's own order carries the semantics `ref2va` needs — see
            // [`crate::reference`] on why order is not incidental, and [`request_references`] for
            // why the video one is `ReferenceVideo` and not `VideoClip` or `VideoSync`.
            //
            // `VideoClip` is deliberately **not** advertised. Nothing here reads it, and leaving it
            // in the allowlist would let an in-context clip through to a model that has no in-context
            // clip mechanism; default-deny turns it into the typed `Error::Unsupported` instead.
            conditioning: vec![
                ConditioningKind::Keyframe,
                ConditioningKind::Reference,
                ConditioningKind::ReferenceVideo,
                ConditioningKind::ReferenceAudio,
            ],
            // sc-18724. Every attention/FFN projection in the 50-block stack and the 2-block token
            // refiner is an `AdaptableLinear` on **every** tier, so the lightx2v turbo LoRAs fold as
            // forward-time residuals with no dequantization — see [`crate::adapters`], which also
            // owns the per-file alpha resolution those checkpoints need.
            supports_lora: true,
            // LoKr routes through the shared LyCORIS seam on the same host
            // ([`crate::adapters::apply_minimax_h3_adapters`]); no MiniMax-H3 LoKr is published, but
            // the surface is real rather than declared, so this is not advertising a stub.
            supports_lokr: true,
            // The tiers exist and load (sc-17150) — packed offline by `crate::convert` and staged
            // as the [`DIT_COMPONENT`]. `spec.quantize` is reconciled against the staged tier's
            // marker rather than triggering a load-time quantize; see [`reconcile_tier`].
            supported_quants: &[Quant::Q4, Quant::Q8],
            min_size: SPATIAL_STRIDE,
            // The widest edge `crate::keyframe::resolve_canvas_size` can put a picture on, at the
            // 4:1 aspect ceiling. A per-edge cap is NOT the area budget — `CANVAS_MAX_PIXELS` is
            // checked as a product by `resolve_geometry` and still refuses 1536x1536 / 1344x1344,
            // which sit inside this ceiling on both edges. See `MAX_CANVAS_EDGE` (sc-17152).
            max_size: MAX_CANVAS_EDGE,
            max_count: 1,
            // The provider's step bound, advertised rather than hidden (sc-19559). `validate`
            // below still refuses the same counts; this is what makes the bound readable
            // weights-free, which matters most for a model whose text encoder is ~53 GB.
            supported_steps: StepSupport::Range {
                min: 1,
                max: MAX_STEPS,
            },
            mac_only: true,
            // The TE → DiT → VAE phase order is hardcoded in `generate_impl`, not a load-time
            // default a `GenerationMemory` block could switch off, so every generate stages
            // physically whatever residency the request selects. This is why
            // [`crate::memory_strategy`] models the floor as `PhaseEnvelope` — `max(TE, DiT, VAE)`
            // and never the sum, because the three are never co-resident. Independent of
            // `supports_sequential_offload` above: the staging is unconditional, but no *selectable*
            // Sequential control is wired onto the shared residency seam.
            unconditionally_engages_staged_residency: true,
            // No enhancement surface exists in this crate: `enhance_prompt` is ignored, so
            // advertising it would describe a toggle a UI must not present as effective.
            supports_prompt_enhancement: false,
            // The audio surface describes a *selectable* audio request (voice / language / rate),
            // which a video model has none of — the soundtrack rides `GenerationOutput::Video`.
            audio_sample_rates: vec![],
            ..Default::default()
        },
    }
}
