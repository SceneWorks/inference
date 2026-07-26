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
//! * **Post-encode eviction sync (sc-12195, seam-enforced by sc-12453).** Candle launches the text
//!   encode asynchronously on CUDA; the seam drops the text phase at a brace and the heavy loader
//!   reuses the freed allocator pool while encode kernels may still reference it — which
//!   deterministically corrupted FLUX.2-dev Q4 pixels until a `Device::synchronize()` was added at the
//!   end of flux2's encode phase. That fix originally landed in flux2 ONLY, leaving the other five
//!   sequential consumers racy (F-002). [`run_sequential`] now performs the boundary sync itself,
//!   after the encode returns and **before** the text phase drops, so every consumer inherits it
//!   structurally. **This is the single point of enforcement** — providers must NOT re-add local
//!   post-encode syncs (flux2's was removed when the seam took it over).
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
//! [`Residency<Text, Heavy>`] also owns the warm `Resident` pair (sc-12128), so providers construct one
//! policy at load and drive both through the same [`Residency::run`] call. A provider can no longer
//! leave a stale component cache beside the sequential path: the residency value is the sole owner.

use candle_core::Device;
use gen_core::runtime::{CancelFlag, LoadPhase};
use gen_core::{OffloadPolicy, Progress};
use std::sync::{Mutex, OnceLock};

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

/// Resolve the load-spec policy together with the family-wide A/B override.
pub fn effective_offload_policy(requested: OffloadPolicy) -> OffloadPolicy {
    if requested == OffloadPolicy::Sequential || sequential_offload_enabled() {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    }
}

type TextLoader<Text> = Box<dyn Fn() -> Result<Text> + Send + Sync>;
type HeavyLoader<Heavy> = Box<dyn Fn(bool) -> Result<Heavy> + Send + Sync>;
type ResidentLoader<Text, Heavy> = Box<dyn Fn() -> Result<(Text, Heavy)> + Send + Sync>;

struct ResidentPair<Text, Heavy> {
    text: Text,
    heavy: Heavy,
}

struct SequentialLoaders<Text, Heavy> {
    load_text: TextLoader<Text>,
    load_heavy: HeavyLoader<Heavy>,
}

struct LazyResident<Text, Heavy> {
    pair: OnceLock<ResidentPair<Text, Heavy>>,
    loader: ResidentLoader<Text, Heavy>,
    load_lock: Mutex<()>,
}

impl<Text, Heavy> LazyResident<Text, Heavy> {
    fn get(&self) -> Result<&ResidentPair<Text, Heavy>> {
        if self.pair.get().is_none() {
            let _guard = crate::lock_recover(&self.load_lock);
            if self.pair.get().is_none() {
                let (text, heavy) = (self.loader)()?;
                let _ = self.pair.set(ResidentPair { text, heavy });
            }
        }
        Ok(self
            .pair
            .get()
            .expect("resident pair is initialized while holding the load lock"))
    }
}

enum Inner<Text, Heavy> {
    Resident(Box<ResidentPair<Text, Heavy>>),
    LazyResident(Box<LazyResident<Text, Heavy>>),
    Sequential(Box<SequentialLoaders<Text, Heavy>>),
}

/// Shared ownership and scheduling for a provider's phase-A text component and heavy render bundle.
/// The resident arm holds both warm; the sequential arm holds only loaders and rebuilds each phase per
/// generation. Both variants drive the same encode/render closures through [`run`](Self::run).
pub struct Residency<Text, Heavy> {
    inner: Inner<Text, Heavy>,
}

impl<Text, Heavy> Residency<Text, Heavy> {
    pub fn resident(text: Text, heavy: Heavy) -> Self {
        Self {
            inner: Inner::Resident(Box::new(ResidentPair { text, heavy })),
        }
    }

    pub fn sequential(
        load_text: impl Fn() -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Inner::Sequential(Box::new(SequentialLoaders {
                load_text: Box::new(load_text),
                load_heavy: Box::new(load_heavy),
            })),
        }
    }

    /// Build the selected policy once. Resident loads eagerly and asks for request-optional heavy
    /// components (`use_pid = true`) so later warm-cache requests can use them; Sequential defers both
    /// loaders and threads the current request's `use_pid` at [`run`](Self::run).
    pub fn from_policy(
        policy: OffloadPolicy,
        load_text: impl Fn() -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Result<Self> {
        match policy {
            OffloadPolicy::Resident => {
                let text = load_text()?;
                let heavy = load_heavy(true)?;
                Ok(Self::resident(text, heavy))
            }
            OffloadPolicy::Sequential => Ok(Self::sequential(load_text, load_heavy)),
        }
    }

    /// Variant of [`from_policy`](Self::from_policy) for providers whose historical resident loader
    /// builds a shared aggregate that cannot be produced by the two independent sequential loaders.
    pub fn from_policy_with_resident(
        policy: OffloadPolicy,
        load_resident: impl Fn() -> Result<(Text, Heavy)> + Send + Sync + 'static,
        load_text: impl Fn() -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Result<Self> {
        match policy {
            OffloadPolicy::Resident => Ok(Self {
                inner: Inner::LazyResident(Box::new(LazyResident {
                    pair: OnceLock::new(),
                    loader: Box::new(load_resident),
                    load_lock: Mutex::new(()),
                })),
            }),
            OffloadPolicy::Sequential => Ok(Self::sequential(load_text, load_heavy)),
        }
    }

    pub fn is_sequential(&self) -> bool {
        matches!(self.inner, Inner::Sequential(_))
    }

    /// Drive one generation. `device` is the provider's compute device; the `Sequential` arm uses it
    /// for the sc-12195 post-encode eviction sync (see [`run_sequential`]). The resident arms never
    /// drop the text phase, so they have no eviction boundary and do not sync.
    pub fn run<Enc, Out>(
        &self,
        cancel: &CancelFlag,
        device: &Device,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> Result<Enc>,
        render: impl FnOnce(&Heavy, Enc, &mut dyn FnMut(Progress)) -> Result<Out>,
    ) -> Result<Out> {
        check_cancel(cancel)?;
        match &self.inner {
            Inner::Resident(pair) => {
                let enc = encode(&pair.text)?;
                check_cancel(cancel)?;
                render(&pair.heavy, enc, on_progress)
            }
            Inner::LazyResident(lazy) => {
                let pair = lazy.get()?;
                let enc = encode(&pair.text)?;
                check_cancel(cancel)?;
                render(&pair.heavy, enc, on_progress)
            }
            Inner::Sequential(loaders) => run_sequential(
                cancel,
                device,
                on_progress,
                || (loaders.load_text)(),
                encode,
                || (loaders.load_heavy)(use_pid),
                render,
            ),
        }
    }
}

/// Drive one generation through the `Sequential` residency lifecycle: **load text → encode → sync the
/// device → drop text → load heavy → render**, with a cancel check at every stage boundary and a
/// [`Progress::Loading`] emit around each load.
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
/// **Post-encode eviction sync (sc-12195 / sc-12453) — enforced HERE, and only here.** Candle
/// launches the text-encode kernels asynchronously on CUDA, and this function drops the text phase at
/// a brace: without a device sync the heavy loader reuses the freed allocator pool while encode
/// kernels may still reference it. That lifetime race deterministically produced different FLUX.2-dev
/// Q4 pixels in resident vs sequential modes; `CUDA_LAUNCH_BLOCKING=1` restored parity and isolated
/// it (sc-12195). The fix originally landed as a local `device.synchronize()` in flux2's encode phase
/// only, leaving the five other sequential consumers racy (F-002) — so the sync moved into this seam
/// (sc-12453): it runs after `encode` returns and **before** the text phase drops, and every consumer
/// inherits it structurally. CPU and Metal synchronize through the same backend-neutral device
/// operation (cheap there). Providers must not re-add local post-encode syncs.
///
/// **cudarc caveat (epic 10765):** dropping the text phase frees into candle's in-process pool, not
/// back to the driver — peak *allocation demand* falls but `nvidia-smi` resident VRAM will not. An A/B
/// only reads true across two separate processes.
pub fn run_sequential<Text, Heavy, Enc, Out>(
    cancel: &CancelFlag,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> Result<Text>,
    encode: impl FnOnce(&Text) -> Result<Enc>,
    load_heavy: impl FnOnce() -> Result<Heavy>,
    render: impl FnOnce(&Heavy, Enc, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    run_sequential_with_sync(
        cancel,
        on_progress,
        load_text,
        encode,
        || Ok(device.synchronize()?),
        load_heavy,
        render,
    )
}

/// Drive a three-component sequential lifecycle while keeping at most one component resident:
/// load/use/drop phase A, load/use/drop phase B, then load/use/drop phase C.
///
/// Video pipelines commonly need one more residency boundary than [`run_sequential`]: a conditioning
/// encoder (and sometimes a source-image VAE encode), then a denoiser, then a VAE decode. Each
/// component is bound inside its own lexical scope. `sync` runs after a successful load and use before
/// the component drops, and also after either operation fails, so asynchronous accelerator work
/// completes before the allocator pool is reused by the next load or job. When both the primary
/// operation and sync fail, the primary error stays primary.
///
/// The caller owns progress and cancellation policy in the load/use closures because phase labels and
/// cancellation points vary by pipeline. `state` carries only the small tensors and metadata that must
/// survive component release.
#[allow(clippy::too_many_arguments)]
pub fn run_three_stage_sequential<A, B, C, St, Out>(
    state: &mut St,
    load_a: impl FnOnce(&mut St) -> Result<A>,
    use_a: impl FnOnce(&A, &mut St) -> Result<()>,
    load_b: impl FnOnce(&mut St) -> Result<B>,
    use_b: impl FnOnce(&B, &mut St) -> Result<()>,
    load_c: impl FnOnce(&mut St) -> Result<C>,
    use_c: impl FnOnce(&C, &mut St) -> Result<Out>,
    mut sync: impl FnMut() -> Result<()>,
) -> Result<Out> {
    {
        let phase_a = match load_a(state) {
            Ok(phase) => phase,
            Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
        };
        let used = use_a(&phase_a, state);
        let synced = sync();
        finish_synchronized_phase(used, synced)?;
    }
    {
        let phase_b = match load_b(state) {
            Ok(phase) => phase,
            Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
        };
        let used = use_b(&phase_b, state);
        let synced = sync();
        finish_synchronized_phase(used, synced)?;
    }
    let phase_c = match load_c(state) {
        Ok(phase) => phase,
        Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
    };
    let used = use_c(&phase_c, state);
    let synced = sync();
    finish_synchronized_phase(used, synced)
}

/// Combine a phase's use result with its mandatory pre-drop synchronization. The synchronization is
/// evaluated by the caller before this function is entered, so it cannot be skipped by an early
/// `?`. Preserve the operation/cancellation error when both fail; that is the actionable job result.
fn finish_synchronized_phase<T>(used: Result<T>, synced: Result<()>) -> Result<T> {
    match used {
        Err(primary) => Err(primary),
        Ok(value) => {
            synced?;
            Ok(value)
        }
    }
}

/// [`run_sequential`] with the sc-12195 post-encode boundary sync injected as a closure, so the
/// ordering contract — encode → sync → text drop → heavy load — is pinnable by tests without a
/// mockable GPU device. Private on purpose: production callers must go through [`run_sequential`],
/// which wires the sync to [`Device::synchronize`].
fn run_sequential_with_sync<Text, Heavy, Enc, Out>(
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> Result<Text>,
    encode: impl FnOnce(&Text) -> Result<Enc>,
    sync_encode_boundary: impl FnOnce() -> Result<()>,
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
        let enc = encode(&text)?;
        // sc-12195: the encode kernels are async — the device MUST drain before `text` frees at the
        // brace below, or the heavy load reuses the freed pool under in-flight kernels.
        sync_encode_boundary()?;
        enc
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
    use std::cell::{Cell, RefCell};
    use std::sync::{Arc, Mutex};

    /// A tripped flag short-circuits before ANY loader runs — the F-173 property that makes a
    /// cancelled staged request cheap.
    #[test]
    fn cancelled_before_start_loads_nothing() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let loaded = RefCell::new(false);

        let out: Result<()> = run_sequential(
            &cancel,
            &Device::Cpu,
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
        assert!(
            !*loaded.borrow(),
            "no loader may run for a cancelled request"
        );
    }

    /// Cancelling during the encode is caught at the NEXT boundary, so the heavy load never starts —
    /// the boundary that matters most (it is the multi-GB one).
    #[test]
    fn cancelled_during_encode_skips_the_heavy_load() {
        let cancel = CancelFlag::new();
        let heavy_loaded = RefCell::new(false);

        let out: Result<()> = run_sequential(
            &cancel,
            &Device::Cpu,
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
        assert!(
            !*heavy_loaded.borrow(),
            "heavy load must not start after a cancel"
        );
    }

    /// Both loads announce themselves (F-179), in phase order, before the render runs — the worker
    /// turns these into the "text encoder" / "render components" job status.
    #[test]
    fn both_load_phases_emit_progress_in_order() {
        let cancel = CancelFlag::new();
        let mut phases = Vec::new();

        let out: Result<u8> = run_sequential(
            &cancel,
            &Device::Cpu,
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
            &Device::Cpu,
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

    /// The sc-12195 eviction sync (sc-12453): the seam's post-encode boundary sync runs AFTER the
    /// encode returns and BEFORE the text phase drops (and therefore before the heavy load). Pinned
    /// through the injectable-sync twin of [`run_sequential`], whose public wrapper wires the hook to
    /// `Device::synchronize` — encode kernels are async, so a sync that ran after the drop (or not at
    /// all) would let the heavy loader reuse freed allocations under in-flight kernels.
    #[test]
    fn boundary_sync_runs_after_encode_and_before_the_text_phase_drops() {
        struct Witness<'a>(&'a RefCell<Vec<&'static str>>);
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push("text-dropped");
            }
        }

        let cancel = CancelFlag::new();
        let log = RefCell::new(Vec::new());

        let out: Result<()> = run_sequential_with_sync(
            &cancel,
            &mut |_| {},
            || Ok(Witness(&log)),
            |_: &Witness| {
                log.borrow_mut().push("encode");
                Ok(())
            },
            || {
                log.borrow_mut().push("sync");
                Ok(())
            },
            || {
                log.borrow_mut().push("heavy-load");
                Ok(())
            },
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(()),
        );

        assert!(out.is_ok());
        assert_eq!(
            *log.borrow(),
            vec!["encode", "sync", "text-dropped", "heavy-load"],
            "the boundary sync must run after the encode and before the text phase drops"
        );
    }

    /// The reusable video lifecycle never co-resides phase components: each used component is synced
    /// and dropped before the next load begins. A provider that regresses to an all-resident aggregate
    /// makes this witness fail.
    #[test]
    fn three_stage_lifecycle_loads_uses_syncs_and_drops_in_order() {
        struct Witness<'a> {
            name: &'static str,
            log: &'a RefCell<Vec<&'static str>>,
        }
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.log.borrow_mut().push(match self.name {
                    "a" => "drop-a",
                    "b" => "drop-b",
                    "c" => "drop-c",
                    _ => unreachable!(),
                });
            }
        }

        let log = RefCell::new(Vec::new());
        let mut state = ();
        let out = run_three_stage_sequential(
            &mut state,
            |_| {
                log.borrow_mut().push("load-a");
                Ok(Witness {
                    name: "a",
                    log: &log,
                })
            },
            |_, _| {
                log.borrow_mut().push("use-a");
                Ok(())
            },
            |_| {
                log.borrow_mut().push("load-b");
                Ok(Witness {
                    name: "b",
                    log: &log,
                })
            },
            |_, _| {
                log.borrow_mut().push("use-b");
                Ok(())
            },
            |_| {
                log.borrow_mut().push("load-c");
                Ok(Witness {
                    name: "c",
                    log: &log,
                })
            },
            |_, _| {
                log.borrow_mut().push("use-c");
                Ok(7)
            },
            || {
                log.borrow_mut().push("sync");
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(out, 7);
        assert_eq!(
            *log.borrow(),
            vec![
                "load-a", "use-a", "sync", "drop-a", "load-b", "use-b", "sync", "drop-b", "load-c",
                "use-c", "sync", "drop-c"
            ]
        );
    }

    /// A phase that fails or is cancelled still drains the device before its component drops. This
    /// is the worker-health contract: the next job must not reuse allocator memory under queued work.
    #[test]
    fn three_stage_failures_sync_before_drop_and_preserve_the_primary_error() {
        struct Witness<'a> {
            name: &'static str,
            log: &'a RefCell<Vec<&'static str>>,
        }
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.log.borrow_mut().push(match self.name {
                    "a" => "drop-a",
                    "b" => "drop-b",
                    "c" => "drop-c",
                    _ => unreachable!(),
                });
            }
        }

        for failing in ["a", "b", "c"] {
            let log = RefCell::new(Vec::new());
            let syncs = Cell::new(0usize);
            let mut state = ();
            let out: Result<()> = run_three_stage_sequential(
                &mut state,
                |_| {
                    log.borrow_mut().push("load-a");
                    Ok(Witness {
                        name: "a",
                        log: &log,
                    })
                },
                |_, _| {
                    log.borrow_mut().push("use-a");
                    if failing == "a" {
                        Err(CandleError::Msg("primary-a".into()))
                    } else {
                        Ok(())
                    }
                },
                |_| {
                    log.borrow_mut().push("load-b");
                    Ok(Witness {
                        name: "b",
                        log: &log,
                    })
                },
                |_, _| {
                    log.borrow_mut().push("use-b");
                    if failing == "b" {
                        Err(CandleError::Canceled)
                    } else {
                        Ok(())
                    }
                },
                |_| {
                    log.borrow_mut().push("load-c");
                    Ok(Witness {
                        name: "c",
                        log: &log,
                    })
                },
                |_, _| {
                    log.borrow_mut().push("use-c");
                    Err(if failing == "c" {
                        CandleError::Msg("primary-c".into())
                    } else {
                        unreachable!()
                    })
                },
                || {
                    log.borrow_mut().push("sync");
                    let sync = syncs.get() + 1;
                    syncs.set(sync);
                    if failing == "c" && sync == 3 {
                        Err(CandleError::Msg("secondary-sync".into()))
                    } else {
                        Ok(())
                    }
                },
            );

            match failing {
                "a" => {
                    assert!(matches!(out, Err(CandleError::Msg(ref msg)) if msg == "primary-a"));
                    assert_eq!(*log.borrow(), vec!["load-a", "use-a", "sync", "drop-a"]);
                }
                "b" => {
                    assert!(matches!(out, Err(CandleError::Canceled)));
                    assert_eq!(
                        *log.borrow(),
                        vec![
                            "load-a", "use-a", "sync", "drop-a", "load-b", "use-b", "sync",
                            "drop-b"
                        ]
                    );
                }
                "c" => {
                    assert!(matches!(out, Err(CandleError::Msg(ref msg)) if msg == "primary-c"));
                    assert_eq!(
                        *log.borrow(),
                        vec![
                            "load-a", "use-a", "sync", "drop-a", "load-b", "use-b", "sync",
                            "drop-b", "load-c", "use-c", "sync", "drop-c"
                        ]
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    /// A loader can fail after queueing only part of a component's host-to-device copies. Every phase
    /// must drain that partial upload before returning the load error, or the next job can reuse its
    /// allocator blocks while copies are still in flight.
    #[test]
    fn three_stage_load_failures_sync_and_preserve_the_primary_error() {
        for failing in ["a", "b", "c"] {
            let log = RefCell::new(Vec::new());
            let mut state = ();
            let out: Result<()> = run_three_stage_sequential(
                &mut state,
                |_| {
                    log.borrow_mut().push("load-a");
                    if failing == "a" {
                        Err(CandleError::Msg("primary-a-load".into()))
                    } else {
                        Ok(())
                    }
                },
                |_, _| {
                    log.borrow_mut().push("use-a");
                    Ok(())
                },
                |_| {
                    log.borrow_mut().push("load-b");
                    if failing == "b" {
                        Err(CandleError::Canceled)
                    } else {
                        Ok(())
                    }
                },
                |_, _| {
                    log.borrow_mut().push("use-b");
                    Ok(())
                },
                |_| {
                    log.borrow_mut().push("load-c");
                    if failing == "c" {
                        Err(CandleError::Msg("primary-c-load".into()))
                    } else {
                        Ok(())
                    }
                },
                |_, _| {
                    log.borrow_mut().push("use-c");
                    Ok(())
                },
                || {
                    let last = log.borrow().last().copied();
                    log.borrow_mut().push("sync");
                    if matches!(
                        (failing, last),
                        ("a", Some("load-a")) | ("b", Some("load-b")) | ("c", Some("load-c"))
                    ) {
                        Err(CandleError::Msg("secondary-sync".into()))
                    } else {
                        Ok(())
                    }
                },
            );

            match failing {
                "a" => {
                    assert!(
                        matches!(out, Err(CandleError::Msg(ref msg)) if msg == "primary-a-load")
                    );
                    assert_eq!(*log.borrow(), vec!["load-a", "sync"]);
                }
                "b" => {
                    assert!(matches!(out, Err(CandleError::Canceled)));
                    assert_eq!(
                        *log.borrow(),
                        vec!["load-a", "use-a", "sync", "load-b", "sync"]
                    );
                }
                "c" => {
                    assert!(
                        matches!(out, Err(CandleError::Msg(ref msg)) if msg == "primary-c-load")
                    );
                    assert_eq!(
                        *log.borrow(),
                        vec![
                            "load-a", "use-a", "sync", "load-b", "use-b", "sync", "load-c", "sync"
                        ]
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    /// A failed boundary sync aborts the run: the heavy load never starts (its allocations would be
    /// exactly the freed-pool reuse the sync exists to prevent), the error propagates, and the text
    /// phase still drops safely via scope drop on the `?` path.
    #[test]
    fn failed_boundary_sync_skips_the_heavy_load() {
        struct Witness<'a>(&'a RefCell<Vec<&'static str>>);
        impl Drop for Witness<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().push("text-dropped");
            }
        }

        let cancel = CancelFlag::new();
        let log = RefCell::new(Vec::new());

        let out: Result<()> = run_sequential_with_sync(
            &cancel,
            &mut |_| {},
            || Ok(Witness(&log)),
            |_: &Witness| Ok(()),
            || Err(CandleError::Msg("device sync failed".into())),
            || {
                log.borrow_mut().push("heavy-load");
                Ok(())
            },
            |_: &(), _: (), _: &mut dyn FnMut(Progress)| Ok(()),
        );

        assert!(matches!(out, Err(CandleError::Msg(_))));
        assert_eq!(
            *log.borrow(),
            vec!["text-dropped"],
            "a sync failure must drop the text phase and skip the heavy load"
        );
    }

    #[test]
    fn from_policy_sequential_defers_and_reloads_each_run() {
        let loads = Arc::new(Mutex::new(Vec::new()));
        let text_loads = Arc::clone(&loads);
        let heavy_loads = Arc::clone(&loads);
        let residency = Residency::from_policy(
            OffloadPolicy::Sequential,
            move || {
                crate::lock_recover(&text_loads).push("text");
                Ok(2u8)
            },
            move |use_pid| {
                heavy_loads
                    .lock()
                    .unwrap()
                    .push(if use_pid { "heavy+pid" } else { "heavy" });
                Ok(3u8)
            },
        )
        .unwrap();

        assert!(residency.is_sequential());
        assert!(crate::lock_recover(&loads).is_empty());
        let out = residency
            .run(
                &CancelFlag::new(),
                &Device::Cpu,
                false,
                &mut |_| {},
                |text| Ok(*text + 1),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
            .unwrap();
        assert_eq!(out, 6);
        assert_eq!(*crate::lock_recover(&loads), vec!["text", "heavy"]);
    }

    #[test]
    fn from_policy_resident_loads_once_with_pid_and_reuses_pair() {
        let loads = Arc::new(Mutex::new(Vec::new()));
        let text_loads = Arc::clone(&loads);
        let heavy_loads = Arc::clone(&loads);
        let residency = Residency::from_policy(
            OffloadPolicy::Resident,
            move || {
                crate::lock_recover(&text_loads).push("text");
                Ok(4u8)
            },
            move |use_pid| {
                heavy_loads
                    .lock()
                    .unwrap()
                    .push(if use_pid { "heavy+pid" } else { "heavy" });
                Ok(5u8)
            },
        )
        .unwrap();

        assert!(!residency.is_sequential());
        assert_eq!(*crate::lock_recover(&loads), vec!["text", "heavy+pid"]);
        for _ in 0..2 {
            let out = residency
                .run(
                    &CancelFlag::new(),
                    &Device::Cpu,
                    false,
                    &mut |_| {},
                    |text| Ok(*text),
                    |heavy, encoded, _| Ok(*heavy + encoded),
                )
                .unwrap();
            assert_eq!(out, 9);
        }
        assert_eq!(*crate::lock_recover(&loads), vec!["text", "heavy+pid"]);
    }

    #[test]
    fn custom_resident_loader_is_lazy_and_cached() {
        let loads = Arc::new(Mutex::new(0usize));
        let resident_loads = Arc::clone(&loads);
        let residency = Residency::from_policy_with_resident(
            OffloadPolicy::Resident,
            move || {
                *crate::lock_recover(&resident_loads) += 1;
                Ok((7u8, 8u8))
            },
            || Ok(0u8),
            |_| Ok(0u8),
        )
        .unwrap();

        assert_eq!(*crate::lock_recover(&loads), 0);
        for _ in 0..2 {
            let out = residency
                .run(
                    &CancelFlag::new(),
                    &Device::Cpu,
                    false,
                    &mut |_| {},
                    |text| Ok(*text),
                    |heavy, encoded, _| Ok(*heavy + encoded),
                )
                .unwrap();
            assert_eq!(out, 15);
        }
        assert_eq!(*crate::lock_recover(&loads), 1);
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
