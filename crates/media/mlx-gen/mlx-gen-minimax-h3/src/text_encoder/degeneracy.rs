//! Runtime degeneracy screening for the conditioning tensors this crate hands the DiT (sc-17153).
//!
//! # Why a runtime check exists on a shipped path
//!
//! The **bf16 dense** H3 text encoder intermittently returns an **all-zero forward**: `max|out|` and
//! `sum|out|` both exactly `0.0`, correct shapes, no Metal error, no allocation anomaly. Measured
//! incidence was **4 sightings in ~30 instrumented forwards (~13%)** on real weights. Nothing
//! downstream notices — the DiT happily denoises against a zero context and returns a video with no
//! relationship to the prompt, which is indistinguishable from "the model is bad at this prompt"
//! unless someone thinks to inspect the conditioning.
//!
//! The mechanism is **below this crate** and, as of 2026-08-20, **unattributed**. Each of the
//! obvious explanations was disproven by counter-measurement:
//!
//! - **Not window-size-specific.** Reversing the window order `[50, 10, 5, 1]` moved the zero to
//!   window 50 rather than keeping it on a fixed window.
//! - **Not deferred/windowing-specific.** 2 of the 4 sightings were plain RESIDENT forwards, one on
//!   a path that never calls [`MiniMaxH3TextEncoder::from_dir_deferred`](super::MiniMaxH3TextEncoder::from_dir_deferred).
//! - **Not warm/cold page cache.** 12/12 warm trials were correct, and a deliberately cold run at
//!   ~60 MB free pages was 4/4 correct.
//! - **Not tier-general.** Zero sightings across ~12 packed q4/q8 arms — it is specific to the
//!   **bf16 dense** encoder, whose token table and projections take a different loader path
//!   ([`mlx_gen::quant::embedding`] returns [`TokenEmbedding::Dense`](mlx_gen::nn::TokenEmbedding)
//!   rather than the packed triple).
//!
//! Because the mechanism is unattributed, the only thing this crate can honestly do is **notice and
//! refuse**. The screen is deliberately *not* a retry: a retry would return a render the user cannot
//! tell apart from a good one on the ~87% of attempts that succeed, and it would erase the incidence
//! data the investigation still needs. See [`Self`](self) and `tests/te_tier_real_weights.rs`.
//!
//! # Why the defect is typed here and not in [`mlx_gen::Error`]
//!
//! [`mlx_gen::Error`] bridges 1:1 to [`gen_core::Error`](mlx_gen::gen_core::Error) in **both**
//! directions, and every typed variant over there exists because some consumer *branches* on it —
//! `Canceled` for the worker, `Unsupported` for candle gating, `GeometryRefused` to carry a verified
//! alternative. A degeneracy refusal has no such consumer by design: nothing may retry it, degrade
//! it, or substitute for it. So the defect is typed *here*, where in-crate callers and tests can
//! match the variant instead of a string, and crosses the crate boundary as
//! [`Error::Msg`] carrying the actionable text — no contract churn, no
//! speculative enum arm on a shared seam.

use mlx_rs::ops::{abs, max, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// What is wrong with a conditioning tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditioningDefect {
    /// Every element is exactly zero — the sc-17153 hazard.
    AllZero,
    /// At least one element is NaN or infinite.
    NonFinite,
}

impl ConditioningDefect {
    /// The clause that names the observation, for the refusal text.
    fn describe(self) -> &'static str {
        match self {
            Self::AllZero => "every element is exactly zero",
            Self::NonFinite => "the tensor holds NaN or infinite values",
        }
    }
}

/// A conditioning tensor that failed the screen, with the two magnitudes the sc-17153 investigation
/// records so a report from the field is directly comparable to its trial log.
#[derive(Debug, Clone, PartialEq)]
pub struct DegenerateConditioning {
    /// Which producer emitted it — the conditioning path, e.g. `"t2va"` or `"vision tower"`.
    pub producer: &'static str,
    /// What is wrong.
    pub defect: ConditioningDefect,
    /// The tensor's shape, so a reader can tell a degenerate value apart from a degenerate shape.
    pub shape: Vec<i32>,
    /// `max|x|` over every element.
    pub max_abs: f32,
    /// `sum|x|` over every element.
    pub sum_abs: f32,
}

impl std::fmt::Display for DegenerateConditioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "minimax-h3 te ({}): refusing to render from a degenerate conditioning tensor — {} at \
             shape {:?} (max|out| {:e}, sum|out| {:e}). The DiT would have denoised against it and \
             returned video unrelated to the prompt. This is NOT retried on purpose: an all-zero \
             context is the intermittent bf16-DENSE text-encoder zeroing tracked by sc-17153 (~13% \
             of forwards; mechanism unattributed as of 2026-08-20), it has never been observed on \
             the q4/q8 packed tiers, so re-running on a packed text-encoder tier is the available \
             workaround — and a silent retry would hide the incidence data that investigation still \
             needs. See crates/media/mlx-gen/mlx-gen-minimax-h3/src/text_encoder/degeneracy.rs.",
            self.producer,
            self.defect.describe(),
            self.shape,
            self.max_abs,
            self.sum_abs,
        )
    }
}

impl std::error::Error for DegenerateConditioning {}

impl From<DegenerateConditioning> for Error {
    fn from(d: DegenerateConditioning) -> Self {
        Error::Msg(d.to_string())
    }
}

/// Screen one conditioning tensor, returning the defect if it is degenerate.
///
/// # Cost
///
/// **Two reductions over one `abs` temporary** — `max|x|` and `sum|x|` — and two 4-byte scalar
/// reads. Nothing of the tensor is copied to the host: only the two reduced scalars are cast to f32
/// and `item`'d, so this is `O(elements)` device bandwidth with no `O(elements)` host traffic. For
/// the widest realistic context (`[1, 2048, 5120]` bf16 = 21 MB) that is ~42 MB of device reads
/// against a **62 GB** encoder forward — under 0.07‰ of the weight traffic the forward it screens
/// already paid, and far below the noise floor of the forward's own timing.
///
/// The screen does force evaluation, because a lazy MLX graph has no value to inspect. That is not
/// added work: [`crate::model`]'s conditioning stages already call `mlx_rs::transforms::eval` on the
/// context immediately after the forward, precisely so the encoder can be released before the DiT
/// allocates. The screen only moves that same `eval` a few lines earlier.
///
/// # Why both magnitudes, honestly
///
/// `sum|x|` earns its place as the **reported** diagnostic: the sc-17153 trial log is written as
/// `max|out| … sum|out| …`, so a refusal from the field lines up with it without conversion.
///
/// It is *also* in the non-finite test, but measured on the pinned MLX/Metal build it is
/// **redundant there**: `max` already propagates NaN, and dropping `|| !sum_abs.is_finite()` reds
/// nothing (verified by mutation). It is kept because NaN propagation through a `max` reduction is
/// not a guarantee this crate can hold MLX to across a pin bump or a different backend — and
/// `sum` propagating NaN *is* arithmetic. `max_propagates_nan_on_the_pinned_backend` pins the
/// current behaviour so a bump that changes it shows up as a test result rather than as silence.
/// Do not read the `sum_abs` term as load-bearing detection on this build; it is portability
/// insurance over a value that has to be computed for the message anyway.
pub fn inspect_conditioning(
    producer: &'static str,
    x: &Array,
) -> Result<Option<DegenerateConditioning>> {
    let magnitude = abs(x)?;
    // `try_item`, never `item`: `Array::item` is `try_item().unwrap()` in the pinned mlx-rs
    // (`mlx-rs/src/array/mod.rs:309-311`), and it is `try_item` that runs the `eval`. A screen on a
    // shipped render path, reached with ~53 GB resident, is exactly where an allocation failure is
    // plausible — this must return `Err` there, not panic the worker.
    let scalar = |a: Array| -> Result<f32> { Ok(a.as_dtype(Dtype::Float32)?.try_item::<f32>()?) };
    let max_abs = scalar(max(&magnitude, None)?)?;
    let sum_abs = scalar(sum(&magnitude, None)?)?;

    let defect = if !max_abs.is_finite() || !sum_abs.is_finite() {
        ConditioningDefect::NonFinite
    } else if max_abs == 0.0 {
        ConditioningDefect::AllZero
    } else {
        return Ok(None);
    };

    Ok(Some(DegenerateConditioning {
        producer,
        defect,
        shape: x.shape().to_vec(),
        max_abs,
        sum_abs,
    }))
}

/// [`inspect_conditioning`], as a refusal — the form every shipped conditioning producer calls.
pub(crate) fn refuse_if_degenerate(producer: &'static str, x: &Array) -> Result<()> {
    match inspect_conditioning(producer, x)? {
        Some(degenerate) => Err(degenerate.into()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(values: &[f32]) -> Array {
        Array::from_slice(values, &[1, 2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// **The mutation target.** An all-zero context — the shape of the sc-17153 hazard — must be
    /// refused, not returned. Delete the `refuse_if_degenerate` call from
    /// [`MiniMaxH3TextEncoder::forward`](super::super::MiniMaxH3TextEncoder::forward) and the
    /// `forward_refuses_an_all_zero_context` arm in the encoder's own tests reds; delete the
    /// `max_abs == 0.0` arm above and this one reds.
    #[test]
    fn an_all_zero_context_is_an_all_zero_defect() {
        let d = inspect_conditioning("t2va", &ctx(&[0.0; 4]))
            .unwrap()
            .expect("an all-zero context must be reported degenerate");
        assert_eq!(d.defect, ConditioningDefect::AllZero);
        assert_eq!(d.max_abs, 0.0);
        assert_eq!(d.sum_abs, 0.0);
        assert_eq!(d.shape, vec![1, 2, 2]);
        assert_eq!(d.producer, "t2va");
    }

    /// A healthy context passes. Without this the screen could be `Some(_)` unconditionally and the
    /// refusal test above would still be green — a guard that refuses everything is not a guard.
    #[test]
    fn a_healthy_context_is_not_degenerate() {
        assert!(inspect_conditioning("t2va", &ctx(&[0.0, -0.5, 0.0, 2.0]))
            .unwrap()
            .is_none());
    }

    /// A single non-zero element is enough — the screen tests the whole tensor, not a corner of it.
    /// A reduction accidentally taken over one axis would leave this `Some(AllZero)`.
    #[test]
    fn one_live_element_saves_an_otherwise_zero_context() {
        assert!(
            inspect_conditioning("t2va", &ctx(&[0.0, 0.0, 0.0, 1.0]))
                .unwrap()
                .is_none(),
            "a whole-tensor reduction cannot call this all-zero"
        );
    }

    /// NaN and infinity are the other way a context can be unusable, and `sum|x|` is what catches
    /// NaN — a `max` reduction's NaN behaviour is implementation-defined, so this arm proves the
    /// second reduction is load-bearing rather than decorative.
    #[test]
    fn non_finite_values_are_their_own_defect() {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let d = inspect_conditioning("t2va", &ctx(&[1.0, poison, 3.0, 4.0]))
                .unwrap()
                .unwrap_or_else(|| panic!("{poison} must be reported degenerate"));
            assert_eq!(d.defect, ConditioningDefect::NonFinite, "{poison}");
        }
    }

    /// **Non-finite detection at a size where the reduction is actually multi-pass.**
    ///
    /// The 4-element arm above is a single-pass reduce and proves nothing about the kernel a real
    /// `[1, s, 5120]` context uses. This builds a `[1, 512, 5120]` bf16 tensor — 2.6 M elements, the
    /// shape of an actual conditioning tensor — with **one** poisoned element buried in the middle,
    /// and requires the defect to still be reported. A tiled reduction that lost the poisoned tile,
    /// or that only checked a prefix, reds here and nowhere else.
    #[test]
    fn non_finite_survives_a_multi_pass_reduction_at_context_size() {
        let (rows, width) = (512usize, 5120usize);
        for (name, poison) in [
            ("NaN", f32::NAN),
            ("inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let mut values = vec![0.25f32; rows * width];
            values[rows * width / 2 + 17] = poison;
            let big = Array::from_slice(&values, &[1, rows as i32, width as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let d = inspect_conditioning("t2va", &big)
                .unwrap()
                .unwrap_or_else(|| panic!("{name} at context size must be reported degenerate"));
            assert_eq!(d.defect, ConditioningDefect::NonFinite, "{name}");
            assert_eq!(d.shape, vec![1, 512, 5120], "{name}");
        }
    }

    /// **The assumption behind the `sum_abs` term, pinned as an observation rather than asserted as
    /// a fact.**
    ///
    /// On the pinned MLX/Metal build a `max` reduction *does* propagate NaN, which is why dropping
    /// `|| !sum_abs.is_finite()` from [`inspect_conditioning`] currently reds nothing. That is a
    /// property of this backend, not a guarantee in MLX's contract.
    ///
    /// If a pin bump flips it, this arm reds and the reviewer learns two things at once: the
    /// behaviour changed, and the `sum_abs` term stopped being redundant and started being the only
    /// thing catching NaN. Either outcome is information; silence would not be.
    #[test]
    fn max_propagates_nan_on_the_pinned_backend() {
        let poisoned = ctx(&[1.0, f32::NAN, 3.0, 4.0]);
        let peak = max(abs(&poisoned).unwrap(), None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .try_item::<f32>()
            .unwrap();
        assert!(
            peak.is_nan(),
            "max stopped propagating NaN (got {peak}) — the `sum_abs` term in \
             `inspect_conditioning` is now load-bearing, not redundant. Update its docs."
        );
    }

    /// A healthy tensor of the same real size must still pass — otherwise the arm above would be
    /// green for a screen that calls every large context degenerate.
    #[test]
    fn a_healthy_context_sized_tensor_is_not_degenerate() {
        let big = Array::from_slice(&vec![0.25f32; 512 * 5120], &[1, 512, 5120])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert!(inspect_conditioning("t2va", &big).unwrap().is_none());
    }

    /// The refusal text has to do four jobs at once: say what was observed, say that the refusal is
    /// deliberate, give the operator something to do, and lead the next engineer to the
    /// investigation. Pinned so a later reword cannot quietly drop one of them.
    #[test]
    fn the_refusal_names_the_observation_the_workaround_and_the_investigation() {
        let e = refuse_if_degenerate("t2va", &ctx(&[0.0; 4]))
            .expect_err("an all-zero context must refuse")
            .to_string();
        assert!(e.contains("every element is exactly zero"), "{e}");
        assert!(e.contains("max|out| 0e0"), "{e}");
        assert!(e.contains("NOT retried"), "{e}");
        assert!(e.contains("packed text-encoder tier"), "{e}");
        assert!(e.contains("sc-17153"), "{e}");
        assert!(e.contains("degeneracy.rs"), "{e}");
    }

    /// The defect crosses the crate boundary as a message-carrying `Error`, not as a bare
    /// "MLX op failed" — the whole point is that an operator can read it.
    #[test]
    fn the_defect_converts_to_a_message_carrying_error() {
        let d = inspect_conditioning("vision tower", &ctx(&[0.0; 4]))
            .unwrap()
            .unwrap();
        match Error::from(d.clone()) {
            Error::Msg(m) => assert_eq!(m, d.to_string()),
            other => panic!("degeneracy must stay readable, got {other:?}"),
        }
    }
}
