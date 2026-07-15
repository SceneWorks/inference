//! Shared component-residency seam for the candle engines (epic 10765 Phase 1c, sc-12089) — the
//! candle counterpart of `mlx-gen`'s [`residency`](https://docs.rs/mlx-gen) module (sc-11125).
//!
//! Every candle provider that wires [`OffloadPolicy::Sequential`](gen_core::OffloadPolicy) runs the
//! same lifecycle inside `generate`: load the phase-A text encoder → encode → **drop it** → load the
//! heavy render bundle (DiT + VAE + overlays) → render. The schedule is model-independent; only the
//! two loaders and the encode/render bodies differ. Before this module, flux (sc-10769), flux2
//! (sc-10868) and qwen-image (sc-10867) had each open-coded it, and each copy independently omitted
//! the same two things — which is exactly the drift mlx-gen's seam was created to stop:
//!
//! * **Stage-boundary cancellation (F-173).** The `Resident` path amortizes its load behind the
//!   components cache, so a request reaches the per-step cancel gate almost immediately. The
//!   `Sequential` path puts a multi-GB, multi-second load *inside* `generate`, ahead of the first
//!   cancellable step — so a cancelled request used to pay the whole preamble. [`run_sequential`]
//!   checks `cancel` at every stage boundary: before the encode, before the heavy load, and after it.
//!
//! * **Load progress (F-179).** [`Progress::Loading`] exists in the contract *for this path* — its
//!   gen-core doc states that without it "the UI would freeze silently while a component streams from
//!   disk". The consumer already renders it (the SceneWorks worker maps it to a `Loading` job event
//!   with "text encoder" / "render components" status), but no candle engine emitted it, so a staged
//!   run showed the user nothing for the duration of both loads. [`run_sequential`] emits it around
//!   each phase.
//!
//! A third gap — loading the optional PiD student (+ its multi-GB caption encoder) on a request that
//! never asked for PiD (F-177) — is not fixable here, because only the provider knows what its heavy
//! bundle contains. The seam's shape is what makes it *visible*: `load_heavy` is a closure the
//! provider builds per generate, so it can read `req.use_pid` and skip the overlay. mlx-gen threads
//! the flag through the seam itself (`load_heavy(use_pid)`); candle providers close over it.
//!
//! **Not ported from mlx-gen:** the `ClearCacheGuard` (F-174) and the `materialize`/`eval` hook. Both
//! are MLX-specific — candle's CUDA allocator has no `empty_cache` (dropping frees into the in-process
//! pool, epic 10765's cudarc caveat) and candle evaluates eagerly, so Rust's own scope-based drop is
//! the whole cleanup story and it already runs on the `?` early-return path.
//!
//! **Still open:** this seam covers the `Sequential` schedule only. mlx-gen's [`Residency<Text, Heavy>`]
//! also unifies the `Resident` path into one type, so a provider drives both policies through a single
//! `run`. Candle's providers still branch (`if sequential { … } else { … }`) and hold their own
//! `Mutex<Option<Components>>` cache. Unifying that is the natural follow-on; it wants its own story,
//! since it touches every wired engine's cache handling.

use gen_core::runtime::{CancelFlag, LoadPhase};
use gen_core::Progress;

use crate::{CandleError, Result};

/// The env var every candle engine reads to force the sequential-residency path, independent of
/// [`LoadSpec::offload_policy`](gen_core::LoadSpec). Shared family-wide on purpose (sc-10769): one A/B
/// runner drives every candle engine with a single export.
pub const OFFLOAD_ENV: &str = "CANDLE_GEN_OFFLOAD";

/// Whether the sequential-residency path is force-enabled by env (epic 10765 Phase 1c).
///
/// Reads [`OFFLOAD_ENV`]: `sequential` (case-insensitive, surrounding whitespace ignored) selects the
/// phased load→encode→drop path regardless of `LoadSpec::offload_policy`; unset or any other value
/// defers to the spec (in production the worker's fit-gate sets the policy). This is the seam the
/// two-process GPU A/B harnesses drive.
///
/// Lives here rather than in each provider (sc-12089): flux, flux2 and qwen-image had grown three
/// byte-identical private copies of this function, so the "shared family-wide" contract the var name
/// encodes was implemented three separate times — and a change to the accepted spelling would have had
/// to find all of them, with no compiler help for the one it missed.
pub fn sequential_offload_enabled() -> bool {
    std::env::var(OFFLOAD_ENV)
        .map(|value| value.trim().eq_ignore_ascii_case("sequential"))
        .unwrap_or(false)
}

/// Return [`CandleError::Canceled`] if `cancel` has been tripped, else `Ok(())`.
///
/// The typed variant matters: the [`From`] bridge lifts it to [`gen_core::Error::Canceled`], which the
/// worker and the gen-core conformance suite key off (sc-4481). A stringified `Msg` would read as a
/// backend failure.
pub fn check_cancel(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }
    Ok(())
}

/// Drive one generation through the `Sequential` residency lifecycle: **load text → encode → drop text
/// → load heavy → render**, with a cancel check at every stage boundary and a [`Progress::Loading`]
/// emit around each load.
///
/// The text phase is scoped so it drops before `load_heavy` runs — that ordering is the entire point,
/// and keeping it here rather than in each provider is what stops it from being re-derived (and
/// re-broken) per engine. `encode`'s product is moved out of the scope, so it must not borrow `Text`.
///
/// `Heavy` is whatever the provider needs downstream of the encode — commonly the DiT + VAE bundle, or
/// a tuple of it with an extra component (e.g. krea's img2img VAE encoder). Building it inside
/// `load_heavy` rather than passing it in is what keeps peak at `max(text, heavy)`.
///
/// **F-177:** `load_heavy` is a closure, so a provider that carries an optional overlay (PiD, control)
/// should close over the request and skip loading what this request will not use — under `Sequential`
/// that load is paid per generate and held resident through the whole denoise.
///
/// **cudarc caveat (epic 10765):** dropping the text phase frees into candle's in-process pool, not
/// back to the driver — peak *allocation demand* falls but `nvidia-smi` resident VRAM will not. An A/B
/// only reads true across two separate processes.
pub fn run_sequential<Text, Heavy, Enc, Out>(
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> Result<Text>,
    encode: impl FnOnce(&Text) -> Result<Enc>,
    load_heavy: impl FnOnce() -> Result<Heavy>,
    render: impl FnOnce(&Heavy, Enc, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    // F-173: a request cancelled before the load preamble returns now, not after two multi-GB loads.
    check_cancel(cancel)?;

    // ── Phase A: load the text encoder, encode, and DROP it at the brace. `enc` is moved out; `text`
    // frees here on the success path and on any `?` early return inside the block (Rust scope drop —
    // candle has no `clear_cache()` to sequence after it, unlike the mlx-gen twin).
    let enc = {
        // F-179: the UI has nothing else to show for the duration of this load.
        on_progress(Progress::Loading(LoadPhase::TextEncoder));
        let text = load_text()?;
        encode(&text)?
    };

    // F-173: before the multi-GB heavy load — the longest uninterruptible stretch of the path.
    check_cancel(cancel)?;
    // F-179: the biggest silent gap — the DiT + VAE (+ overlays) streaming from disk.
    on_progress(Progress::Loading(LoadPhase::Renderer));
    let heavy = load_heavy()?;
    // F-173: after the load, before the render commits to the denoise loop (which has its own
    // per-step gate).
    check_cancel(cancel)?;

    render(&heavy, enc, on_progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A tripped flag short-circuits before ANY loader runs — the F-173 property that makes a
    /// cancelled staged request cheap.
    #[test]
    fn cancelled_before_start_loads_nothing() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let loaded = RefCell::new(false);

        let out: Result<()> = run_sequential(
            &cancel,
            &mut |_| {},
            || {
                *loaded.borrow_mut() = true;
                Ok(())
            },
            |_: &()| Ok(()),
            || {
                *loaded.borrow_mut() = true;
                Ok(())
            },
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(()),
        );

        assert!(matches!(out, Err(CandleError::Canceled)));
        assert!(!*loaded.borrow(), "no loader may run for a cancelled request");
    }

    /// Cancelling during the encode is caught at the NEXT boundary, so the heavy load never starts —
    /// the boundary that matters most (it is the multi-GB one).
    #[test]
    fn cancelled_during_encode_skips_the_heavy_load() {
        let cancel = CancelFlag::new();
        let heavy_loaded = RefCell::new(false);

        let out: Result<()> = run_sequential(
            &cancel,
            &mut |_| {},
            || Ok(()),
            |_: &()| {
                cancel.cancel();
                Ok(())
            },
            || {
                *heavy_loaded.borrow_mut() = true;
                Ok(())
            },
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(()),
        );

        assert!(matches!(out, Err(CandleError::Canceled)));
        assert!(!*heavy_loaded.borrow(), "heavy load must not start after a cancel");
    }

    /// Both loads announce themselves (F-179), in phase order, before the render runs — the worker
    /// turns these into the "text encoder" / "render components" job status.
    #[test]
    fn both_load_phases_emit_progress_in_order() {
        let cancel = CancelFlag::new();
        let mut phases = Vec::new();

        let out: Result<u8> = run_sequential(
            &cancel,
            &mut |p| {
                if let Progress::Loading(phase) = p {
                    phases.push(phase);
                }
            },
            || Ok(()),
            |_: &()| Ok(()),
            || Ok(()),
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(7),
        );

        assert_eq!(out.unwrap(), 7);
        assert_eq!(phases, vec![LoadPhase::TextEncoder, LoadPhase::Renderer]);
    }

    /// The text phase is dropped BEFORE the heavy load starts — the property the whole path exists
    /// for. Asserted through a drop-order witness rather than by reading VRAM.
    #[test]
    fn text_phase_drops_before_the_heavy_load() {
        struct Witness<'a>(&'a RefCell<Vec<&'static str>>);
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push("text-dropped");
            }
        }

        let cancel = CancelFlag::new();
        let log = RefCell::new(Vec::new());

        let out: Result<()> = run_sequential(
            &cancel,
            &mut |_| {},
            || Ok(Witness(&log)),
            |_: &Witness| Ok(()),
            || {
                log.borrow_mut().push("heavy-load");
                Ok(())
            },
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(()),
        );

        assert!(out.is_ok());
        assert_eq!(*log.borrow(), vec!["text-dropped", "heavy-load"]);
    }

    /// The env reader: case- and whitespace-insensitive on `sequential`, false for everything else.
    /// Serial by construction — `.cargo/config.toml` force-pins `RUST_TEST_THREADS=1` (F-160).
    #[test]
    fn offload_env_reads_sequential_case_insensitively() {
        let prior = std::env::var(OFFLOAD_ENV).ok();

        std::env::set_var(OFFLOAD_ENV, "  SeQuEnTiAl  ");
        assert!(sequential_offload_enabled());
        std::env::set_var(OFFLOAD_ENV, "resident");
        assert!(!sequential_offload_enabled());
        std::env::set_var(OFFLOAD_ENV, "");
        assert!(!sequential_offload_enabled());
        std::env::remove_var(OFFLOAD_ENV);
        assert!(!sequential_offload_enabled());

        match prior {
            Some(v) => std::env::set_var(OFFLOAD_ENV, v),
            None => std::env::remove_var(OFFLOAD_ENV),
        }
    }
}
