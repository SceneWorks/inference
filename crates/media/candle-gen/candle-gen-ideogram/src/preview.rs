//! Ideogram 4 per-step latent previews (epic 16948, sc-16955) — the **patch-major** sibling of
//! [`candle_gen_flux2::preview`].
//!
//! ## What is reused, and what is not
//!
//! Ideogram loads the FLUX.2 VAE verbatim ([`Flux2Vae`](candle_gen_flux2::vae::Flux2Vae)), so it
//! shares the 32-channel latent space and therefore the epic-16624 fit: the projection itself is
//! [`candle_gen_flux2::preview::project_raw_latents`], called through, never copied. What it does
//! **not** share is the route to that latent. FLUX.2 packs the 128 transformer channels as
//! `(c=32, ph=2, pw=2)`; Ideogram's DiT packs them as `(ph=2, pw=2, c=32)`. The two orders are the
//! same 128 numbers permuted, so a FLUX.2-shaped recovery applied here would de-normalize against a
//! permuted stat vector and unpatchify along the wrong axes — and produce a plausible picture rather
//! than an error. That is the trap `mlx-gen-flux2/src/preview.rs` names, and the reason
//! `Flux2Vae::bn_stats` / `Flux2Vae::decode_latent` exist at all.
//!
//! `crate::pipeline::raw_latent` therefore owns the de-normalize + `(ph,pw,c)` unpatchify, and this
//! module is the thin composition of it with the shared fit. Because `crate::pipeline::decode` calls
//! that same function, a per-step frame and the finished image are recovered by one implementation.
//!
//! ## The bespoke loop
//!
//! Ideogram is the epic's first genuine bespoke consumer: it drives **no** shared sampler at all
//! (`git grep run_flow_sampler -- candle-gen-ideogram` is empty), so it emits through
//! [`candle_gen::preview::emit_preview_at`] from inside its own denoise loop rather than by handing a
//! hook to a driver. `candle-gen-catalog`'s route inventory declares that as `Denoise::Bespoke` and
//! recognises the direct emission call, which is what lets the crate be verified rather than
//! hard-failed for having nothing to hook.
//!
//! Two consequences of owning the loop, both handled in [`crate::pipeline`]:
//!
//! * **Numbering is step-index keyed**, not σ keyed. The `LogitNormalSchedule` is inverted (a larger
//!   σ is *cleaner*) and evaluated per step rather than indexed out of a descending array, so
//!   [`candle_gen::preview::PreviewCounter::with_steps`] — the shape the σ-less SCM driver uses — is
//!   the right counter.
//! * **The total is `num_run`, not `steps`.** An edit runs only the lowest `floor(steps·strength)`
//!   positions, so a strength-0.5 img2img previews half as many frames and each one reports that
//!   smaller total.
//!
//! ## What a frame can and cannot contain
//!
//! * **CFG never reaches a frame.** The quality route's unconditional DiT runs inside the loop body
//!   and is blended into one velocity (`pos·gw + neg·(1−gw)`) before `z` is advanced; turbo has no
//!   unconditional branch at all. `z` is `[1, num_img, 128]` at every step.
//! * **Text tokens never reach a frame.** `text_z_padding` is concatenated onto `z` to build the DiT
//!   input and the result is narrowed back to the image tokens, so the running latent is the image
//!   grid alone.
//! * **The inpaint source never reaches a frame** any more than it reaches the render: the mask blend
//!   writes the re-noised source into `z` itself, so a frame shows exactly what the trajectory holds.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, Result};

/// The latent channel count the reused fit is defined over, re-exported from the owner so this crate
/// cannot drift from it by restating a number.
pub use candle_gen_flux2::preview::PREVIEW_LATENT_CHANNELS;

/// The packed channel count Ideogram's DiT carries — the same 128 as FLUX.2, in a different order.
pub use candle_gen_flux2::preview::PACKED_LATENT_CHANNELS;

/// Project Ideogram's packed running latent `[1, grid_h·grid_w, 128]` to a latent-resolution RGB8
/// preview: `crate::pipeline::raw_latent` (bn de-normalize + `(ph,pw,c)` unpatchify) followed by the
/// shared 32-channel fit.
///
/// The frame is `2·grid_h × 2·grid_w` — the raw VAE latent resolution, twice the token grid — because
/// the unpatchify runs before the projection.
///
/// Errors on any layout that is not this route's packed shape. The caller's frame is then lost and
/// swallowed by [`candle_gen::preview::emit_preview_at`], the intended decorative-failure behaviour.
pub fn project_packed_tokens(
    comps: &crate::pipeline::Components,
    z: &Tensor,
    grid_h: usize,
    grid_w: usize,
) -> Result<Image> {
    check_packed_layout(z, grid_h, grid_w)?;
    let raw = crate::pipeline::raw_latent(comps, z, grid_h, grid_w)?;
    candle_gen_flux2::preview::project_raw_latents(&raw)
}

/// Staged twin of [`project_packed_tokens`]. Only the two small BN tensors are retained from the
/// optional-encode phase; decoder weights are released before the DiT opens.
pub(crate) fn project_packed_tokens_with_stats(
    bn_std: &Tensor,
    bn_mean: &Tensor,
    z: &Tensor,
    grid_h: usize,
    grid_w: usize,
) -> Result<Image> {
    check_packed_layout(z, grid_h, grid_w)?;
    let raw = crate::pipeline::raw_latent_with_stats(bn_std, bn_mean, z, grid_h, grid_w)?;
    candle_gen_flux2::preview::project_raw_latents(&raw)
}

/// Reject a packed token sequence that is not this render's grid.
///
/// Checked before `crate::pipeline::raw_latent`'s reshape so the message names the grid and the
/// caller loses one decorative frame rather than seeing a raw candle shape error. It catches a
/// sequence length or channel width that does not describe the declared grid — a transposition of the
/// same length is not detectable from the sequence alone, and is ruled out at the call site instead by
/// `denoise` deriving the grid exactly as `decode` does.
fn check_packed_layout(z: &Tensor, grid_h: usize, grid_w: usize) -> Result<()> {
    let dims = z.dims();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != grid_h * grid_w
        || dims[2] != PACKED_LATENT_CHANNELS
    {
        return Err(CandleError::Msg(format!(
            "ideogram preview latent must have shape [1, {}, {PACKED_LATENT_CHANNELS}] for a \
             {grid_h}x{grid_w} token grid, got {dims:?}",
            grid_h * grid_w
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::{DType, Device, Tensor};

    use super::*;

    /// Ideogram introduces no fit of its own: the constants it projects with are `candle-gen-flux2`'s,
    /// over the 32-channel FLUX.2 space. A crate-local copy would fork one latent space into two colour
    /// maps. (The needles are assembled at compile time so this scan does not match its own source.)
    #[test]
    fn ideogram_reuses_the_flux2_fit_rather_than_restating_it() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 32);
        assert_eq!(PACKED_LATENT_CHANNELS, 128);
        for (file, source) in [
            ("pipeline.rs", include_str!("pipeline.rs")),
            ("lib.rs", include_str!("lib.rs")),
        ] {
            assert!(
                !source.contains(concat!("RGB_", "FACTORS"))
                    && !source.contains(concat!("RGB_", "BIAS")),
                "{file} holds an Ideogram-local copy of the FLUX.2 fit"
            );
        }
    }

    /// The layout gate rejects everything that is not this render's packed grid, and lets the grid
    /// through. Weights-free — it runs before any VAE is touched.
    #[test]
    fn the_layout_gate_accepts_the_grid_and_rejects_the_rest() {
        let z = Tensor::zeros((1, 12, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&z, 3, 4).is_ok());
        for (h, w) in [(4usize, 4usize), (3, 3), (1, 11)] {
            let error = check_packed_layout(&z, h, w).unwrap_err().to_string();
            assert!(error.contains("token grid"), "{h}x{w}: {error}");
        }
        let narrow = Tensor::zeros((1, 12, 64), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&narrow, 3, 4).is_err());
        let batched = Tensor::zeros((2, 12, 128), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&batched, 3, 4).is_err());
        let spatial = Tensor::zeros((1, 128, 3, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(check_packed_layout(&spatial, 3, 4).is_err());
    }

    /// The `(ph,pw,c)` unpatchify is **not** FLUX.2's `(c,ph,pw)` one, pinned as a numeric difference
    /// rather than as prose. Both orders are applied to the same 128-channel cell and the results are
    /// asserted different — this is the whole reason `crate::pipeline::raw_latent` exists, so it must
    /// fail if the two ever became interchangeable.
    ///
    /// Reproduces the two folds directly (identity bn stats) rather than through a `Flux2Vae`, whose
    /// ~55M-parameter decoder a channel-order row has no use for.
    #[test]
    fn the_patch_major_order_differs_from_the_flux2_channel_major_one() {
        let device = Device::Cpu;
        let (grid_h, grid_w) = (1usize, 1usize);
        let values: Vec<f32> = (0..PACKED_LATENT_CHANNELS).map(|c| c as f32).collect();

        // Ideogram: [1, L, 128] -> (gh, gw, ph, pw, c) -> [1, c, gh*2, gw*2].
        let packed = Tensor::from_vec(values.clone(), (1, grid_h * grid_w, 128), &device).unwrap();
        let ideogram = packed
            .reshape((1, grid_h, grid_w, 2, 2, 32))
            .unwrap()
            .permute((0, 5, 1, 3, 2, 4))
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((1, 32, grid_h * 2, grid_w * 2))
            .unwrap();

        // FLUX.2: the same numbers on the [1, 128, gh, gw] grid through the canonical unpatchify.
        let grid = Tensor::from_vec(values, (1, 128, grid_h, grid_w), &device).unwrap();
        let ones = Tensor::ones((1, 128, 1, 1), DType::F32, &device).unwrap();
        let zeros = Tensor::zeros((1, 128, 1, 1), DType::F32, &device).unwrap();
        let flux2 = candle_gen_flux2::vae::raw_latent_from_packed(&grid, &ones, &zeros).unwrap();

        assert_eq!(ideogram.dims(), flux2.dims());
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_ne!(
            flat(&ideogram),
            flat(&flux2),
            "if the two packing orders ever agreed, crate::pipeline::raw_latent would be dead code \
             and this crate should call the flux2 helper instead"
        );
        // Ideogram's channel c comes from packed channels {ph*64 + pw*32 + c}; FLUX.2's from
        // {c*4 + ph*2 + pw}. Spot-check channel 0's four patch cells under each order.
        assert_eq!(flat(&ideogram)[..4], [0.0, 32.0, 64.0, 96.0]);
        assert_eq!(flat(&flux2)[..4], [0.0, 1.0, 2.0, 3.0]);
    }

    /// The bespoke loop's emission contract, pinned against `pipeline.rs`'s own source: exactly one
    /// direct emission call, keyed on the step index, over the `num_run` total.
    ///
    /// This crate drives no shared sampler, so there is no call site whose `preview` argument could be
    /// inspected — the source-level fact available here is that the loop emits, once, in the shape the
    /// catalog's `Denoise::Bespoke` row declares. (Needles are assembled at compile time so this scan
    /// does not match its own source.)
    #[test]
    fn the_bespoke_denoise_loop_emits_exactly_once_per_step() {
        let source = include_str!("pipeline.rs");
        assert_eq!(
            source.matches(concat!("emit_preview", "_at(")).count(),
            1,
            "the bespoke loop must make exactly one direct emission call"
        );
        assert!(
            source.contains(concat!("PreviewCounter::", "with_steps(num_run)")),
            "the counter must be step-index keyed over num_run, not over the requested step count"
        );
        assert!(
            source.contains("num_run - 1 - i"),
            "the emission must be keyed on the loop's 0-based position"
        );
        // No shared sampler anywhere in the crate: the fact that makes this crate Denoise::Bespoke.
        for (file, code) in [
            ("pipeline.rs", source),
            ("lib.rs", include_str!("lib.rs")),
            ("loader.rs", include_str!("loader.rs")),
            ("adapters.rs", include_str!("adapters.rs")),
            ("scheduler.rs", include_str!("scheduler.rs")),
        ] {
            assert!(
                !code.contains(concat!("run_flow_", "sampler(")),
                "{file} drives a shared sampler — this crate is declared Denoise::Bespoke"
            );
        }
    }

    /// Both registered Ideogram routes advertise the flag. Weights-free: descriptors only.
    #[test]
    fn both_registered_ideogram_routes_advertise_preview_support() {
        for descriptor in [crate::descriptor(), crate::descriptor_turbo()] {
            assert!(
                descriptor.capabilities.supports_preview,
                "{} must advertise preview support",
                descriptor.id
            );
        }
    }
}
