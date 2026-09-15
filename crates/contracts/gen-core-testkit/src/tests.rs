//! The testkit verifying itself: a configurable in-crate stub generator drives each conformance
//! check, and one deliberately-broken variant per check proves the check actually fires (the
//! sc-4481 AC). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use super::*;
use gen_core::registry::ModelRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image,
    Modality, ModelDescriptor, Progress,
};
use std::cell::Cell;

/// The registered stub id (round-trips through the explicit fixture registry below).
const STUB_ID: &str = "testkit_stub";
/// A stub id that is deliberately NOT registered — exercises the registry-check failure path.
const UNREG_ID: &str = "testkit_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all of them; each broken-stub test
/// flips exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the capability floor (vs. rubber-stamping every request).
    honest_validate: bool,
    /// Emits a `Progress::Step` per denoise iteration.
    emit_progress: bool,
    /// Number of `Progress::Decoding` events emitted after the step loop (contract requires exactly 1).
    decoding_events: u32,
    /// Emit `Step.current` up to `2*total` — the F-050 multi-eval-sampler overrun (>100%).
    overrun_steps: bool,
    /// Stop emitting `Step` at `total - 1` while still advertising `total` — the F-030 frozen-below-total
    /// (PiD early-stop) shape.
    freeze_below_total: bool,
    /// Checks `CancelFlag` at each step boundary and bails.
    honor_cancel: bool,
    /// On cancel, returns the typed `Error::Canceled` (vs. a stringified `Error::Msg`).
    typed_cancel: bool,
    /// Output pixels depend only on the seed (vs. drifting per call).
    deterministic: bool,
    /// sc-17418: at `guidance = 1.0` (CFG off) the negative prompt is inert — the correct behaviour,
    /// since the combine `uncond + 1·(cond − uncond)` reduces to `cond`. The broken variant folds the
    /// negative into the output, which is exactly what an engine that narrows its conditioning to the
    /// WRONG row does: it renders the negative prompt instead of the prompt.
    cfg_off_ignores_negative: bool,
    /// sc-17418: `generate` succeeds at `guidance = 1.0`. The broken variant is the literal sc-14195
    /// shape — `validate` waves the request through and the engine then dies mid-denoise.
    cfg_off_generates: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_progress: true,
            decoding_events: 1,
            overrun_steps: false,
            freeze_below_total: false,
            honor_cancel: true,
            typed_cancel: true,
            deterministic: true,
            cfg_off_ignores_negative: true,
            cfg_off_generates: true,
        }
    }
}

struct Stub {
    desc: ModelDescriptor,
    behavior: Behavior,
    /// When set, the honest `validate` runs the **size-skipping** floor
    /// (`Capabilities::validate_request_skip_size`) instead of the full `validate_request` — the
    /// audio-lane / auto-size stance where `width`/`height` are not range-checked (sc-13705).
    skip_size: bool,
    /// sc-19502: when set, the honest `validate` runs the floor against a Capabilities whose
    /// `supported_steps` has been CLEARED — so the descriptor advertises a fixed schedule that
    /// `validate` never enforces. That is precisely the `mlx-gen-ltx` defect this story fixed
    /// (advertised != enforced), and `check_validate_honesty` must catch it.
    unenforced_steps: bool,
    /// Per-instance call counter — the nondeterministic variant fills pixels from this.
    runs: Cell<u32>,
}

fn stub_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::Reference],
        min_size: 64,
        max_size: 512,
        max_count: 4,
        ..Default::default()
    }
}

/// Capabilities for a **CFG-capable** stub (sc-17418): advertises the guidance + negative-prompt
/// axes, which is what puts a model under [`check_cfg_off_render`]'s contract at all. The plain
/// [`stub_caps`] leaves both `false`, so that stub is (correctly) skipped by the check.
fn guided_stub_caps() -> Capabilities {
    Capabilities {
        supports_guidance: true,
        supports_negative_prompt: true,
        ..stub_caps()
    }
}

fn guided_stub_desc(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        capabilities: guided_stub_caps(),
        ..stub_desc(id)
    }
}

/// Capabilities for an **audio-lane** stub (sc-13705): `Modality::Audio` providers advertise no size
/// bound (`min_size`/`max_size` are the unused 0 — sc-13314) and validate through the size-skipping
/// floor, so `width`/`height` are not part of their contract.
fn audio_stub_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::Reference],
        min_size: 0,
        max_size: 0,
        max_count: 4,
        ..Default::default()
    }
}

fn stub_desc(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id,
        family: "testkit",
        backend: "stub",
        modality: Modality::Image,
        capabilities: stub_caps(),
    }
}

fn audio_stub_desc(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id,
        family: "testkit",
        backend: "stub",
        modality: Modality::Audio,
        capabilities: audio_stub_caps(),
    }
}

impl Stub {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
            skip_size: false,
            unenforced_steps: false,
            runs: Cell::new(0),
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Generator> {
        Box::new(Self::new(id, behavior))
    }

    /// A **CFG-capable** stub (sc-17418): advertises guidance + negative prompt, so
    /// [`check_cfg_off_render`] actually engages instead of skipping.
    fn guided(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: guided_stub_desc(id),
            behavior,
            skip_size: false,
            unenforced_steps: false,
            runs: Cell::new(0),
        }
    }

    /// An **audio-lane** stub (sc-13705): `Modality::Audio`, no size bound (`max_size` 0), validating
    /// through the size-skipping floor — the shape whose 64×64 (== `max_size` 0 + 64) request the
    /// generic oversize probe must NOT flag, because `width`/`height` are meaningless for audio.
    fn audio(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: audio_stub_desc(id),
            behavior,
            skip_size: true,
            unenforced_steps: false,
            runs: Cell::new(0),
        }
    }

    fn boxed_audio(id: &'static str, behavior: Behavior) -> Box<dyn Generator> {
        Box::new(Self::audio(id, behavior))
    }

    /// sc-19502: a stub advertising a fixed 8-step schedule. `enforced = false` is the
    /// `mlx-gen-ltx` defect shape — the descriptor claims the constraint, `validate` never applies
    /// it, and the engine silently renders its baked schedule for whatever count arrives.
    fn fixed_schedule(id: &'static str, enforced: bool) -> Self {
        let mut desc = stub_desc(id);
        desc.capabilities.supported_steps = StepSupport::Exact(vec![8]);
        Self {
            desc,
            behavior: Behavior::good(),
            skip_size: false,
            unenforced_steps: !enforced,
            runs: Cell::new(0),
        }
    }

    /// sc-19559: a stub advertising a step RANGE rather than an exact menu — SVD's shape. The
    /// `enforced = false` twin is the same defect as [`Stub::fixed_schedule`]'s: the descriptor
    /// declares the ceiling and `validate` never applies it, so a caller's over-ceiling `steps`
    /// reaches the engine.
    fn bounded_range(id: &'static str, enforced: bool) -> Self {
        let mut desc = stub_desc(id);
        desc.capabilities.supported_steps = StepSupport::Range { min: 1, max: 8 };
        Self {
            desc,
            behavior: Behavior::good(),
            skip_size: false,
            unenforced_steps: !enforced,
            runs: Cell::new(0),
        }
    }

    /// An **image** stub that (wrongly) validates through the size-skipping floor while still
    /// advertising a real `max_size`. Used to prove the generic oversize probe STILL fires for
    /// size-relevant (non-audio) providers after the audio exemption (sc-13705) — the max_size check
    /// must not be weakened for images.
    fn image_skipping_size(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
            skip_size: true,
            unenforced_steps: false,
            runs: Cell::new(0),
        }
    }
}

impl Generator for Stub {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        if !self.behavior.honest_validate {
            return Ok(());
        }
        let caps = &self.desc.capabilities;
        if self.unenforced_steps {
            // Advertises the schedule, enforces everything BUT it (sc-19502).
            let mut lax = caps.clone();
            lax.supported_steps = StepSupport::Unconstrained;
            return lax.validate_request(self.desc.id, req);
        }
        if self.skip_size {
            // The audio-lane / auto-size floor: every shared check except the width/height range.
            caps.validate_request_skip_size(self.desc.id, req)
        } else {
            caps.validate_request(self.desc.id, req)
        }
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        let total = req.steps.unwrap_or(2);
        let run = self.runs.get();
        self.runs.set(run + 1);
        // How many Step events actually get emitted: `total` (good), `2*total` (F-050 overrun),
        // or `total - 1` (F-030 frozen below its advertised total).
        let emit_max = if self.behavior.overrun_steps {
            total.saturating_mul(2)
        } else if self.behavior.freeze_below_total {
            total.saturating_sub(1)
        } else {
            total
        };
        for i in 1..=emit_max {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                return Err(if self.behavior.typed_cancel {
                    Error::Canceled
                } else {
                    Error::Msg("generation cancelled".into())
                });
            }
            if self.behavior.emit_progress {
                on_progress(Progress::Step { current: i, total });
            }
        }
        for _ in 0..self.behavior.decoding_events {
            on_progress(Progress::Decoding);
        }
        // sc-17418, the CFG-off contract. `cfg_off` is "this request has classifier-free guidance
        // switched off" — the fork every engine hand-writes, and the one sc-14195 got wrong.
        let cfg_off = req.guidance.is_some_and(|g| g <= 1.0);
        if cfg_off && !self.behavior.cfg_off_generates {
            // The literal sc-14195 shape: validate() waved it through, the engine dies mid-denoise.
            return Err(Error::Msg(
                "shape mismatch in matmul, lhs: [10, 4096, 64], rhs: [20, 64, 77]".into(),
            ));
        }
        let mut fill = if self.behavior.deterministic {
            req.seed.unwrap_or(0) as u8
        } else {
            run as u8
        };
        // The wrong-row narrow: at CFG-off the negative branch is gone, so a correct engine cannot
        // let the negative prompt reach the output. This variant does, and must be caught.
        if cfg_off && !self.behavior.cfg_off_ignores_negative {
            let neg = req.negative_prompt.as_deref().unwrap_or("");
            fill = fill.wrapping_add(neg.len() as u8).wrapping_add(
                neg.bytes()
                    .fold(0u8, |acc, b| acc.wrapping_mul(31).wrapping_add(b)),
            );
        }
        let img = Image {
            width: req.width,
            height: req.height,
            pixels: vec![fill; req.width as usize * req.height as usize * 3],
        };
        Ok(GenerationOutput::Images(vec![img]))
    }
}

fn stub_descriptor() -> ModelDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Stub::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: ModelRegistration = ModelRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
    footprint: None,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_generator(STUB_REGISTRATION)
        .build()
        .expect("stub registry should build")
}

fn cheap() -> Profile {
    Profile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    conformance(|| Stub::boxed(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let g = Stub::new(STUB_ID, Behavior::good());
    check_validate_honesty(&g, &cheap()).unwrap();
    check_progress(&g, &cheap()).unwrap();
    check_progress_contract(&g, &cheap()).unwrap();
    check_cancellation(&g, &cheap()).unwrap();
    check_precancellation(&g, &cheap()).unwrap();
    check_seed_determinism(&g, &cheap()).unwrap();
    check_registry_roundtrip(&registry(), &g).unwrap();
}

#[test]
fn ignoring_cancel_fails_precancellation_check() {
    // A provider that never consults the flag runs to completion even on an already-cancelled
    // request — the non-denoise-seam class (sc-11128): it returns Ok instead of typed Canceled.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_precancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_precancellation_check() {
    // Honors the flag up front but stringifies the error — must still fail the typed contract.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_precancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_validate_honesty(&g, &cheap()).is_err());
}

#[test]
fn audio_provider_size_exemption_passes_validate_honesty() {
    // sc-13705: an audio-lane provider (Modality::Audio) validates through the size-skipping floor
    // and advertises no size bound (max_size 0 — sc-13314), because width/height are meaningless for
    // audio. The generic image-shaped honesty check must therefore NOT probe a max_size+64 (== 64x64)
    // oversize rejection for it: the provider legitimately accepts that request. Before the fix this
    // failed with "a 64x64 request (above max_size 0) was accepted by validate()" — the exact
    // false-inconsistency between the audio size exemption and the shared testkit.
    let g = Stub::audio(STUB_ID, Behavior::good());
    check_validate_honesty(&g, &cheap()).unwrap();
    // And the whole generic suite runs green for an audio-lane provider.
    conformance(|| Stub::boxed_audio(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn image_provider_oversize_probe_still_fires_after_audio_exemption() {
    // The audio exemption is keyed on Modality::Audio ONLY: a size-relevant (image/video) provider
    // that skips its size range check must STILL fail the oversize probe, so the max_size assertion
    // stays meaningful for image providers (the sc-13705 fix must not delete/weaken it). This stub is
    // Modality::Image with a real max_size (512) but validates via the size-skipping floor, so a
    // 576x576 (max_size+64) request slips through its validate() — the check must catch that.
    let g = Stub::image_skipping_size(STUB_ID, Behavior::good());
    let err = check_validate_honesty(&g, &cheap()).unwrap_err();
    assert!(err.contains("above max_size"), "got: {err}");
}

/// sc-19502 — a descriptor that ADVERTISES a fixed step schedule but whose `validate` never
/// enforces it must fail the honesty check.
///
/// This is the `mlx-gen-ltx` defect in miniature and the reason the sweep needs its negative half:
/// a lane that ignores `req.steps` entirely satisfies the positive half trivially (every advertised
/// count "validates", because everything validates). Only probing an OFF-menu count catches it.
#[test]
fn advertising_a_fixed_schedule_without_enforcing_it_fails_validate_honesty() {
    // The profile's steps must sit ON the advertised menu, or the harness's first positive check
    // would fail for an honest provider too — the LTX conformance profile already pins 8.
    let profile = Profile {
        steps: 8,
        ..cheap()
    };

    let dishonest = Stub::fixed_schedule(STUB_ID, false);
    let err = check_validate_honesty(&dishonest, &profile)
        .expect_err("advertising a schedule it does not enforce must be caught");
    assert!(
        err.contains("advertised surface") && err.contains("silently ignores"),
        "the failure must name the defect: {err}"
    );

    // …and the honest twin passes, so the check is not simply rejecting every declaring provider.
    check_validate_honesty(&Stub::fixed_schedule(STUB_ID, true), &profile)
        .expect("a provider that enforces what it advertises is honest");
}

/// sc-19559 — the same honesty contract for the RANGE shape. A provider that advertises a ceiling
/// and does not enforce it is the SVD defect the story names: `MAX_STEPS = 200` was real inside
/// the engine but invisible on the surface, and the mirror failure — a surface that claims a
/// bound the engine ignores — is what this catches.
///
/// Deliberately its own test rather than an extra assertion in the exact-menu one above: the two
/// shapes take different arms of `check_validate_honesty`'s probe construction, and a regression
/// in the range arm alone would otherwise hide behind the menu arm still passing.
#[test]
fn advertising_a_step_range_without_enforcing_it_fails_validate_honesty() {
    // Inside the advertised 1..=8, so an honest provider's positive probes all pass.
    let profile = Profile {
        steps: 4,
        ..cheap()
    };

    let dishonest = Stub::bounded_range(STUB_ID, false);
    let err = check_validate_honesty(&dishonest, &profile)
        .expect_err("advertising a ceiling it does not enforce must be caught");
    assert!(
        err.contains("step count 9") && err.contains("Range") && err.contains("silently ignores"),
        "the failure must name the over-ceiling count and the advertised range: {err}"
    );

    check_validate_honesty(&Stub::bounded_range(STUB_ID, true), &profile)
        .expect("a provider that enforces its advertised range is honest");
}

/// sc-19502 — `check_cancellation` must not hand a fixed-schedule provider the profile's headroom
/// `cancel_steps`, which is off its menu and would surface a step-count REJECTION as a cancellation
/// defect. It falls back to the largest advertised count instead.
#[test]
fn cancellation_uses_an_advertised_step_count_for_a_fixed_schedule_provider() {
    let profile = Profile {
        steps: 8,
        // 6 is the default headroom and is deliberately OFF the advertised [8] menu.
        cancel_steps: 6,
        ..cheap()
    };
    let g = Stub::fixed_schedule(STUB_ID, true);
    check_cancellation(&g, &profile)
        .expect("cancellation must run at an advertised count, not the off-menu headroom");
}

#[test]
fn missing_progress_fails_progress_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    assert!(check_progress(&g, &cheap()).is_err());
}

#[test]
fn overrunning_steps_fail_progress_contract() {
    // The F-050 class: a multi-eval sampler double-counts and reports current up to 2*total.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            overrun_steps: true,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("exceeds total"), "got: {err}");
}

#[test]
fn freezing_below_total_fails_progress_contract() {
    // The F-030 class: an early-stopped schedule never reaches its advertised total.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            freeze_below_total: true,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("must reach"), "got: {err}");
}

#[test]
fn missing_decoding_fails_progress_contract() {
    // The F-030 class: the decode stage is invisible because Decoding is never emitted.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            decoding_events: 0,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("emitted 0 times"), "got: {err}");
}

#[test]
fn repeated_decoding_fails_progress_contract() {
    // The F-136/F-162 restarting-bar class: Decoding (or the bar) restarts per output.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            decoding_events: 3,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("emitted 3 times"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("ran to completion"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    // The exact pre-sc-4481 family behavior: stops early but returns Error::Msg, not Canceled.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn nondeterministic_fails_seed_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            deterministic: false,
            ..Behavior::good()
        },
    );
    assert!(check_seed_determinism(&g, &cheap()).is_err());
}

#[test]
fn unregistered_id_fails_registry_check() {
    let g = Stub::new(UNREG_ID, Behavior::good());
    assert!(check_registry_roundtrip(&registry(), &g).is_err());
}

/// sc-17418: a CFG-capable model that serves `guidance = 1.0` correctly — renders, and leaves the
/// negative prompt inert — passes.
#[test]
fn correct_cfg_off_passes() {
    let g = Stub::guided(STUB_ID, Behavior::good());
    assert!(check_cfg_off_render(&g, &cheap()).is_ok());
}

/// sc-17418, the literal sc-14195 shape: `validate` accepts `guidance = 1.0` and the engine then
/// dies mid-denoise on a batch mismatch. The check must fire, and must say so in terms that point
/// at the CFG batch contract rather than just "generate failed".
#[test]
fn cfg_off_that_explodes_fails_check() {
    let g = Stub::guided(
        STUB_ID,
        Behavior {
            cfg_off_generates: false,
            ..Behavior::good()
        },
    );
    let err = check_cfg_off_render(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("validate() accepted guidance = 1.0"),
        "got: {err}"
    );
    assert!(err.contains("shape mismatch"), "got: {err}");
}

/// sc-17418, the mutation a liveness-only check CANNOT catch: the engine narrows its conditioning to
/// the WRONG row, so CFG-off renders the negative prompt. It still returns an image — nothing
/// errors — so only the negative-inertness assertion catches it.
#[test]
fn cfg_off_that_consumes_the_negative_fails_check() {
    let g = Stub::guided(
        STUB_ID,
        Behavior {
            cfg_off_ignores_negative: false,
            ..Behavior::good()
        },
    );
    // It really does render — this is not an error path.
    let mut req = crate::base_request(&cheap());
    req.guidance = Some(1.0);
    req.negative_prompt = Some("anything".into());
    assert!(g.generate(&req, &mut |_| {}).is_ok());

    let err = check_cfg_off_render(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("the negative prompt changed the output"),
        "got: {err}"
    );
    assert!(err.contains("WRONG row"), "got: {err}");
}

/// sc-17418: a model with no guidance axis (the distilled/CFG-free families) is out of scope — the
/// check skips rather than inventing a contract the descriptor never advertised. The plain stub
/// leaves `supports_guidance` false, and would FAIL the broken behaviours above if it were graded.
#[test]
fn cfg_free_model_is_skipped_not_graded() {
    let broken = Behavior {
        cfg_off_generates: false,
        cfg_off_ignores_negative: false,
        ..Behavior::good()
    };
    let g = Stub::new(STUB_ID, broken);
    assert!(!g.descriptor().capabilities.supports_guidance);
    assert!(check_cfg_off_render(&g, &cheap()).is_ok());
    // Mutation guard for the skip itself: the SAME broken behaviour on a guidance-advertising
    // descriptor does fail, so the skip is driven by the capability, not by the check being inert.
    assert!(check_cfg_off_render(&Stub::guided(STUB_ID, broken), &cheap()).is_err());
}

/// sc-17418: honest rejection is a legitimate stance. A model whose `validate` refuses
/// `guidance = 1.0` is not obliged to render it — the check passes without calling `generate`.
#[test]
fn honest_rejection_of_cfg_off_passes() {
    struct Rejecting(ModelDescriptor);
    impl Generator for Rejecting {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.0
        }
        fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
            if req.guidance.is_some_and(|g| g <= 1.0) {
                return Err(Error::Msg("this model requires guidance > 1".into()));
            }
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            panic!("generate() must not be called once validate() has rejected the request");
        }
    }
    let g = Rejecting(guided_stub_desc(STUB_ID));
    assert!(check_cfg_off_render(&g, &cheap()).is_ok());
}

/// The weights-free descriptor sweep (sc-9098, F-009) is clean over the explicit fixture registry.
/// Per-violation firing is unit-tested next to the checks in `gen_core::registry`.
#[test]
fn registry_sweep_passes_for_the_registered_stub() {
    registry_conformance(&registry());
}

/// `check_progress_with` accepts a request-supplied run (the SVD/SeedVR2/renderer shape) and flags
/// a resolved-total mismatch when `expected_total` is pinned.
#[test]
fn progress_with_checks_request_supplied_runs() {
    let g = Stub::new(STUB_ID, Behavior::good());
    let req = GenerationRequest {
        prompt: "a fox".into(),
        width: 128,
        height: 128,
        steps: Some(3),
        seed: Some(7),
        ..Default::default()
    };
    check_progress_with(&g, &req, Some(3)).unwrap();
    check_progress_with(&g, &req, None).unwrap();
    let err = check_progress_with(&g, &req, Some(5)).unwrap_err();
    assert!(err.contains("expected resolved step count"), "got: {err}");
}

// -------------------------------------------------------------------------------------------------
// Named-component load gate (sc-13658) — the `check_component_load_gate` helper exercised against a
// stub loader wired to the *real* gen-core validators (not a mock).
// -------------------------------------------------------------------------------------------------

/// The component ids the component-gate stub declares (mirrors chatterbox's provisional set).
const GATE_REQUIRED: &[&str] = &["perth", "voice_embedding"];

/// A base spec that stages every required component — the positive input the gate removes from /
/// adds to. Paths are placeholders: the stub loader validates components without reading weights.
fn gate_base_spec() -> LoadSpec {
    LoadSpec::new(gen_core::WeightsSource::Dir(std::path::PathBuf::from(
        "/snap",
    )))
    .with_component(
        "perth",
        gen_core::WeightsSource::File(std::path::PathBuf::from("/perth.safetensors")),
    )
    .with_component(
        "voice_embedding",
        gen_core::WeightsSource::File(std::path::PathBuf::from("/voice.safetensors")),
    )
}

/// A **correct** loader: it wires both real validators before building the generator, so a missing
/// required component or an unknown key becomes a load-time error.
fn gate_good_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    gen_core::reject_unknown_components(spec, GATE_REQUIRED, STUB_ID)?;
    for id in GATE_REQUIRED {
        gen_core::require_component(spec, id, STUB_ID, "stub component")?;
    }
    Ok(Stub::boxed(STUB_ID, Behavior::good()))
}

#[test]
fn component_load_gate_passes_for_a_correctly_gated_loader() {
    check_component_load_gate(gate_good_load, &gate_base_spec(), GATE_REQUIRED).unwrap();
}

#[test]
fn component_load_gate_flags_a_loader_that_skips_require_component() {
    // A loader that never calls require_component silently proceeds (the perth mid-render-fetch
    // class) — the gate must catch that a missing required component was accepted.
    fn ungated(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
        Ok(Stub::boxed(STUB_ID, Behavior::good()))
    }
    let err = check_component_load_gate(ungated, &gate_base_spec(), GATE_REQUIRED).unwrap_err();
    assert!(err.contains("must be a load-time error"), "got: {err}");
}

#[test]
fn component_load_gate_flags_a_loader_that_skips_unknown_key_rejection() {
    // A loader that requires its components but never rejects unknown keys silently ignores a stray
    // component — the gate must catch the accepted unknown key.
    fn no_unknown_guard(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
        for id in GATE_REQUIRED {
            gen_core::require_component(spec, id, STUB_ID, "stub component")?;
        }
        Ok(Stub::boxed(STUB_ID, Behavior::good()))
    }
    let err =
        check_component_load_gate(no_unknown_guard, &gate_base_spec(), GATE_REQUIRED).unwrap_err();
    assert!(err.contains("unrecognized component key"), "got: {err}");
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    conformance(
        || {
            Stub::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}

/// A qwen3-shaped encoder, sized like the ones the provider load-gate tests actually build. The
/// dimensions are the point: they are what makes `write_encoder_contract_fixture` a ~7.3 GB writer,
/// which is in turn why the payload has to be a hole rather than bytes.
const SPARSE_FIXTURE_CONTRACT: gen_core::EncoderContract = gen_core::EncoderContract {
    architecture: "qwen3",
    hidden_size: 2560,
    intermediate_size: 9728,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
    output_width: 2560,
    loaded_hidden_layers: 36,
    requires_final_norm: false,
    requires_lm_head: false,
    hidden_activation: "silu",
    attention_dropout: gen_core::EncoderConfigFloat::new(0.0),
    rms_norm_eps: gen_core::EncoderConfigFloat::new(1e-6),
    qk_norm_eps: Some(gen_core::EncoderConfigFloat::new(1e-6)),
    rope_theta: gen_core::EncoderConfigFloat::new(1_000_000.0),
    max_position_embeddings: 40_960,
    attention_bias: gen_core::EncoderConfigBool::Required(false),
    tie_word_embeddings: gen_core::EncoderConfigBool::Required(true),
    tokenizer: gen_core::EncoderTokenizerContract {
        family: "testkit",
        binding: gen_core::EncoderTokenizerBinding::RetainBase,
        artifact_candidates: &["tokenizer.json"],
        required_tokens: &[],
    },
    prompt_executions: &[],
    bos_token_id: Some(151_643),
    eos_token_id: Some(151_645),
    image_token_id: None,
    vision_start_token_id: None,
    vision_end_token_id: None,
    mrope_section: &[],
    mrope_interleaved: None,
    selected_hidden_layers: &[35],
    packing: None,
    dense_storage_dtype_probe: None,
};

/// The fixture's tensor payload must be a **hole**, not written bytes.
///
/// Every gate this fixture feeds *stats* the file, so its logical length has to be the full nominal
/// size — but nothing may read the payload. On APFS and ext4, extending past the end is sparse by
/// construction, so this cost nothing and nobody noticed. NTFS allocates the clusters instead: these
/// fixtures wrote ~7.3 GB apiece (~22 GB per run across the q4/q8/bf16 tiers), filled the Windows
/// CUDA box's system drive twice, and failed a `candle-worker` run with `StorageFull` (os error 112).
///
/// Kills the mutation that matters: dropping the [`mark_sparse`] call from
/// [`write_encoder_contract_fixture_with_quant`] — or moving it *before* the `File::create` that
/// clears the attribute — leaves `FILE_ATTRIBUTE_SPARSE_FILE` unset and this red on Windows. The
/// unflagged control below is what gives that assertion teeth.
///
/// The Linux `contracts` lane still runs the length and zero-payload halves, which pin the semantics
/// the flag must not disturb.
#[test]
fn encoder_contract_fixture_payload_is_a_hole_not_seven_gigabytes() {
    let tmp = tempfile::tempdir().unwrap();
    // Deliberately not named `text_encoder`: that spelling also emits a tokenizer artifact, which is
    // irrelevant here and would only add a second file to reason about.
    let root = tmp.path().join("encoder");
    let expected_headers =
        encoder_contract_fixture_tensor_headers(SPARSE_FIXTURE_CONTRACT, None).unwrap();
    write_encoder_contract_fixture(&root, SPARSE_FIXTURE_CONTRACT).unwrap();

    let weights = root.join("model.safetensors");
    let actual_headers = gen_core::safetensors_path_tensor_headers(&weights).unwrap();
    let by_name = |headers: Vec<gen_core::SafetensorsTensorHeader>| {
        headers
            .into_iter()
            .map(|header| (header.name, (header.dtype, header.shape, header.data_bytes)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(
        by_name(actual_headers),
        by_name(expected_headers),
        "in-memory header facts must be byte-exact with the sparse writer without reading payload"
    );
    let meta = std::fs::metadata(&weights).unwrap();
    assert!(
        meta.len() > 4 << 30,
        "this writer must stay in the multi-GB class or the hole stops being load-bearing: {}",
        meta.len()
    );

    // A hole reads back as zeros. Nothing may depend on whatever the filesystem last left there,
    // which is exactly what flagging the file changes for a reader.
    let mut file = std::fs::File::open(&weights).unwrap();
    let mut declared = [0_u8; 8];
    std::io::Read::read_exact(&mut file, &mut declared).unwrap();
    let header_end = 8 + u64::from_le_bytes(declared);
    assert!(
        header_end < meta.len(),
        "the fixture must have a payload span past its header"
    );
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(meta.len() - 4096)).unwrap();
    let mut tail = [0xAB_u8; 4096];
    std::io::Read::read_exact(&mut file, &mut tail).unwrap();
    assert!(
        tail.iter().all(|&byte| byte == 0),
        "the unwritten payload must read back as zeros"
    );

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        /// `FILE_ATTRIBUTE_SPARSE_FILE`.
        const SPARSE: u32 = 0x0000_0200;

        // The control: the same create-then-extend shape with no flag. If NTFS ever started
        // reporting *this* as sparse, the assertion below would have stopped meaning anything.
        let dense = tmp.path().join("dense.safetensors");
        std::fs::File::create(&dense)
            .unwrap()
            .set_len(1 << 20)
            .unwrap();
        assert_eq!(
            std::fs::metadata(&dense).unwrap().file_attributes() & SPARSE,
            0,
            "control: an unflagged extension must NOT report sparse"
        );

        assert_ne!(
            meta.file_attributes() & SPARSE,
            0,
            "the fixture must carry FILE_ATTRIBUTE_SPARSE_FILE, or NTFS allocates all {} bytes",
            meta.len()
        );
    }
}

#[test]
fn in_memory_encoder_header_facts_reject_dense_only_packing() {
    let error = encoder_contract_fixture_tensor_headers(SPARSE_FIXTURE_CONTRACT, Some(4))
        .unwrap_err()
        .to_string();
    assert!(error.contains("dense-only"), "{error}");
}

/// Copying a fixture must not be how the hole gets materialized.
///
/// `std::fs::copy` reads and writes every byte, so a test that copies a multi-GB fixture to say
/// "another file appeared" pays for the whole payload on NTFS even when the source is a hole —
/// which is precisely how one qwen-image test allocated 26 GB. The copy has to reproduce the header
/// and the logical length onto a flagged destination instead.
///
/// Kills the mutation: swapping [`copy_sparse_fixture`] back to `std::fs::copy` drops
/// `FILE_ATTRIBUTE_SPARSE_FILE` from the destination. The byte assertions here are what pin that the
/// cheaper copy is still a faithful one.
#[test]
fn copying_a_fixture_reproduces_it_without_materializing_the_hole() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("encoder");
    write_encoder_contract_fixture(&root, SPARSE_FIXTURE_CONTRACT).unwrap();
    let source = root.join("model.safetensors");
    let destination = root.join("added.safetensors");
    copy_sparse_fixture(&source, &destination).unwrap();

    let source_meta = std::fs::metadata(&source).unwrap();
    let copy_meta = std::fs::metadata(&destination).unwrap();
    assert_eq!(
        copy_meta.len(),
        source_meta.len(),
        "the copy must stat the same as the original"
    );

    // Header bytes identical, payload still a hole reading as zeros — together that is exactly what
    // `std::fs::copy` of this fixture would have produced.
    let read_head = |path: &std::path::Path| {
        let mut bytes = vec![0_u8; 64 << 10];
        let mut file = std::fs::File::open(path).unwrap();
        std::io::Read::read_exact(&mut file, &mut bytes).unwrap();
        bytes
    };
    assert_eq!(
        read_head(&source),
        read_head(&destination),
        "the copy must carry the original's header"
    );
    let mut file = std::fs::File::open(&destination).unwrap();
    std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(copy_meta.len() - 4096)).unwrap();
    let mut tail = [0xAB_u8; 4096];
    std::io::Read::read_exact(&mut file, &mut tail).unwrap();
    assert!(
        tail.iter().all(|&byte| byte == 0),
        "the copy's payload must read back as zeros"
    );

    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        /// `FILE_ATTRIBUTE_SPARSE_FILE`.
        const SPARSE: u32 = 0x0000_0200;
        assert_ne!(
            copy_meta.file_attributes() & SPARSE,
            0,
            "the copy must be a hole too, or it allocates all {} bytes",
            copy_meta.len()
        );
    }
}

/// A truncated or corrupt fixture must be a typed error, not a panic or a silent empty copy.
#[test]
fn copying_rejects_a_header_that_runs_past_the_end_of_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("truncated.safetensors");
    std::fs::write(&source, u64::MAX.to_le_bytes()).unwrap();
    let error = copy_sparse_fixture(&source, &tmp.path().join("copy.safetensors"))
        .expect_err("a header longer than the file must not be copied");
    assert!(error.to_string().contains("runs past the end"), "{error}");
}

/// [`mark_sparse`] is best effort: a fixture that lands dense still holds exactly the right bytes,
/// so nothing it does may become a test failure. An unwrapped `fsutil` invocation would turn an
/// unwritable path, a locked file, or a runner image without `fsutil` into a failure of every
/// fixture test in the workspace instead of a disk-space regression.
#[test]
fn mark_sparse_tolerates_a_path_that_does_not_exist() {
    let tmp = tempfile::tempdir().unwrap();
    mark_sparse(&tmp.path().join("no-such-fixture.safetensors"));
}
