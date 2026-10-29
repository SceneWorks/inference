//! The SDXL-family generate stage skeleton (sc-27336), shared by `sdxl` and `mlx-gen-kolors`
//! because both drive the same SDXL U-Net and VAE.
//!
//! A registered SDXL-family generate runs four named stages:
//!
//! 1. `prepare_conditioning` — seed-independent, once per request: the family's typed mode plan,
//!    its ControlNet embeddings / IP image tokens, and the VAE-encoded init latent. The inputs
//!    differ per family (dual CLIP vs ChatGLM3, MultiControlNet vs a single pose branch), so each
//!    family keeps its own function under that name.
//! 2. [`FamilyStages::build_init`] — the seeded per-output start. Every global-RNG draw of an
//!    output happens here, after the caller seeded the stream for that output.
//! 3. [`FamilyStages::denoise`] — the mode's denoise assembly, returning the final latent in the
//!    frame the decoder consumes.
//! 4. [`decode_output`] — the VAE (or PiD) decode of one final latent.
//!
//! [`run_output`] fixes the order of stages 2 and 3 and how long their state lives, so the two
//! families cannot drift: whatever `build_init` retains stays alive until the output's latent has
//! been handed on, exactly as the historical inline count loops held their sampler, inpaint blend
//! and noise locals.

use mlx_gen::gen_core::sampling::VpCapturePlan;
use mlx_gen::{CancelFlag, Image, LatentDecoder, Progress, Result};
use mlx_rs::Array;

use crate::vae::Autoencoder;

/// Stages 2 and 3 of an SDXL-family output, implemented by each family's per-request plan.
pub trait FamilyStages {
    /// What the denoise consumes by value (SDXL: the seeded start latent; Kolors: nothing).
    type Start;
    /// What lives for the whole output, through the hand-off of its final latent (SDXL: the
    /// sampler, inpaint blend and drawn noise; Kolors: the drawn noise).
    type Retained;

    /// Draw this output's seeded start. The caller has already seeded the global RNG stream.
    fn build_init(&self, seed: u64) -> Result<(Self::Start, Self::Retained)>;

    /// Run the mode's denoise assembly from `start`, returning the final latent.
    fn denoise(
        &self,
        start: Self::Start,
        retained: &Self::Retained,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array>;
}

/// One output through stages 2 and 3, then `emit` (the family's collect or decode). The retained
/// init state is dropped only after `emit` returns.
pub fn run_output<S: FamilyStages + ?Sized>(
    stages: &S,
    seed: u64,
    on_progress: &mut dyn FnMut(Progress),
    emit: impl FnOnce(Array, &mut dyn FnMut(Progress)) -> Result<()>,
) -> Result<()> {
    let (start, retained) = stages.build_init(seed)?;
    let latent = stages.denoise(start, &retained, seed, on_progress)?;
    emit(latent, on_progress)?;
    drop(retained);
    Ok(())
}

/// Map a captured raw variance-exploding latent (`x0 + σ·ε`) into the PiD student's
/// variance-preserving frame with the capture plan's rescale (`1/√(1+σ²)`). No plan (the clean
/// σ = 0 decode) returns the latent untouched.
pub fn rescale_ve_capture(latent: Array, vp_plan: Option<&VpCapturePlan>) -> Result<Array> {
    Ok(match vp_plan {
        Some(plan) => mlx_rs::ops::multiply(&latent, mlx_gen::array::scalar(plan.rescale))?,
        None => latent,
    })
}

/// The strength tail of a curated σ schedule: `int(steps·strength)` denoise steps ending at σ = 0.
/// A strength that rounds to zero steps leaves only the terminal node. The caller resolves (and,
/// where its family does, clamps) the strength.
pub fn curated_strength_tail(full_sigmas: &[f32], steps: usize, strength: f32) -> Vec<f32> {
    crate::sampling::strength_tail(full_sigmas, steps, strength)
}

/// Stage 4: report [`Progress::Decoding`] and decode one final latent through the VAE, or through
/// `pid` when the request minted a PiD decoder.
pub fn decode_output(
    vae: &Autoencoder,
    latent: &Array,
    pid: Option<&dyn LatentDecoder>,
    tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Image> {
    on_progress(Progress::Decoding);
    crate::pipeline::decode_image_tiled(vae, latent, pid, tiling, Some(cancel))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host-only stage double: records the call order so the skeleton's contract — init, then
    /// denoise, then emit, with the retained state outliving the emit — is pinned without compute.
    struct Recorder<'a> {
        log: &'a std::cell::RefCell<Vec<&'static str>>,
    }

    struct Retained<'a> {
        log: &'a std::cell::RefCell<Vec<&'static str>>,
    }

    impl Drop for Retained<'_> {
        fn drop(&mut self) {
            self.log.borrow_mut().push("drop-retained");
        }
    }

    impl<'a> FamilyStages for Recorder<'a> {
        type Start = ();
        type Retained = Retained<'a>;

        fn build_init(&self, _seed: u64) -> Result<((), Retained<'a>)> {
            self.log.borrow_mut().push("init");
            Ok(((), Retained { log: self.log }))
        }

        fn denoise(
            &self,
            (): (),
            _retained: &Retained<'a>,
            _seed: u64,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<Array> {
            self.log.borrow_mut().push("denoise");
            Err(mlx_gen::Error::Msg("host-only double".into()))
        }
    }

    #[test]
    fn run_output_stops_at_the_first_failing_stage_and_still_drops_retained() {
        let log = std::cell::RefCell::new(Vec::new());
        let stages = Recorder { log: &log };
        let err = run_output(&stages, 7, &mut |_| {}, |_, _| {
            log.borrow_mut().push("emit");
            Ok(())
        })
        .unwrap_err();
        assert!(err.to_string().contains("host-only double"));
        assert_eq!(*log.borrow(), ["init", "denoise", "drop-retained"]);
    }
}
