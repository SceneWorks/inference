//! Shared component-residency seam (epic 10834; sc-11125, consolidating F-173/F-174/F-175/F-177).
//!
//! Every wired image provider that supports [`OffloadPolicy::Sequential`](crate::OffloadPolicy) had
//! grown a near-verbatim copy of the same machinery: a two-variant residency state, an `encode` that
//! loaded the text encoder → encoded → `eval`ed → dropped it → `clear_cache()`, a `load_seq_heavy`,
//! a `heavy()` borrow-resolver with an identical `unreachable!`, and a `was_sequential` cleanup tail.
//! Eight copies (sdxl, z-image, qwen-image ×3, krea ×2, lens) drifted independently — and each of the
//! systematic gaps the 2026-07-11 review named (no stage-boundary cancel, no error-path cache flush,
//! per-generate PiD load) then needed the same fix applied eight times.
//!
//! [`Residency`] hoists that machinery into ONE generic seam, parameterized by the phase-A **`Text`**
//! component (dropped first under `Sequential`) and the heavy **`Heavy`** render bundle (DiT + VAE +
//! any control/PiD overlay). A provider builds one at load time — [`Residency::resident`] holds every
//! component warm; [`Residency::sequential`] captures two per-phase loader closures and holds nothing —
//! and drives a generation through [`Residency::run`], which runs the staged `encode → free text →
//! load heavy → render → free heavy` lifecycle identically for both policies.
//!
//! The seam owns the discipline the copies each re-derived:
//!
//! * **F-175 (redundancy):** the enum, the eval/drop/clear ordering, the borrow resolution, and the
//!   cleanup tail live here once; a provider supplies only the model-specific closures.
//! * **F-173 (cancellation):** [`run`](Residency::run) checks `req.cancel` at every stage boundary —
//!   before the encode, before the heavy load, and after it — so a request cancelled during the (now
//!   multi-GB, multi-second) `Sequential` load phases returns [`Error::Canceled`] promptly instead of
//!   running the whole preamble. The denoise loop keeps its own per-step gate; this covers the seams
//!   between stages the loop never sees.
//! * **F-174 (leak on early exit):** under `Sequential` the text encoder and the heavy bundle are each
//!   freed by an RAII `ClearCacheGuard`, so `clear_cache()` runs on the error/cancel `?`-return path
//!   too — not only on the success tail the copies covered. A cancelled/failed job no longer idles
//!   holding a DiT-sized cache.
//! * **F-177 (wasted PiD load):** the heavy loader closure receives `use_pid`, so a `Sequential`
//!   generate whose request does not use PiD skips loading the PiD student (+ its caption encoder)
//!   entirely instead of loading and holding it unused through denoise.
//!
//! `Resident` behavior is byte-for-byte the pre-seam warm-cache path (no eval, no `clear_cache()`,
//! components loaded once and borrowed) apart from the two cheap stage-boundary cancel checks, which
//! only affect an already-cancelled request (one that produces no output).

use crate::error::{Error, Result};
use crate::runtime::{CancelFlag, LoadPhase, OffloadPolicy, Progress};
use std::sync::{Mutex, MutexGuard};

/// Boxed phase-A loader: rebuilds the `Text` component (text/vision encoder) from the captured load
/// spec. `streamable` selects the request's component form at the point of use rather than at
/// generator construction. `Send + Sync` keeps a `Residency`-holding generator's auto-traits.
type TextLoader<Text> = Box<dyn Fn(bool) -> Result<Text> + Send + Sync>;

/// Boxed heavy-phase loader: rebuilds the `Heavy` render bundle (DiT + VAE + overlays). The `bool` is
/// the request's `use_pid` (F-177); the second is the request's `streamable` component form.
type HeavyLoader<Heavy> = Box<dyn Fn(bool, bool) -> Result<Heavy> + Send + Sync>;

/// The warm-resident pair: the phase-A `Text` component + the `Heavy` render bundle, both held for the
/// whole job and across jobs. `streamable` records the form used to build the pair so a request that
/// needs the other form evicts before reloading instead of mixing resident and streamed weights.
struct ResidentPair<Text, Heavy> {
    text: Text,
    heavy: Heavy,
    streamable: bool,
}

/// The two per-phase loader closures retained for the generator's whole lifetime.
struct SeqLoaders<Text, Heavy> {
    load_text: TextLoader<Text>,
    load_heavy: HeavyLoader<Heavy>,
}

/// A heavy render bundle whose denoise-only component (the DiT) can be **shed** after the denoise
/// phase, leaving a lighter decode bundle (the VAE + any decoder overlay) for a memory-bounded tiled
/// decode (sc-13571, GitHub #1658). Under `Sequential`, freeing the multi-GB DiT *before* the VAE
/// decode drops the decode-phase peak below the DiT+decode-transient sum on a small Mac (the z-image
/// 1024² decode transient alone is ~14 GiB); under `Resident` nothing is shed (the whole bundle stays
/// warm across jobs) and the same decode view is borrowed from it. Providers whose heavy bundle is a
/// DiT + a light decode bundle implement this to opt into [`Residency::run_staged`].
pub trait StagedHeavy {
    /// The owned light (decode-only) bundle that survives the DiT drop — held by the `Sequential`
    /// staged run between the shed and the decode.
    type Light;
    /// A borrowed decode view, produced from the owned [`Light`](Self::Light) (`Sequential`, post-shed)
    /// or from the still-warm `&Self` (`Resident`), so the decode body is written once for both.
    type DecodeView<'a>
    where
        Self: 'a,
        Self::Light: 'a;
    /// Consume the heavy bundle, dropping the DiT and returning the light bundle (`Sequential`).
    fn shed_dit(self) -> Self::Light;
    /// Borrow the decode view from the still-held heavy bundle (`Resident`).
    fn decode_view(&self) -> Self::DecodeView<'_>;
    /// Borrow the decode view from the owned light bundle (`Sequential`, after [`shed_dit`](Self::shed_dit)).
    fn light_view(light: &Self::Light) -> Self::DecodeView<'_>;
}

/// The shared component-residency strategy for a provider (epic 10834; see the module docs).
///
/// `Text` is the phase-A component dropped first under `Sequential` (a text or vision-language
/// encoder, or a tuple of them). `Heavy` is the render bundle — everything but the text encoder (the
/// DiT/U-Net, the VAE, and any ControlNet / PiD overlay).
pub struct Residency<Text, Heavy> {
    loaders: SeqLoaders<Text, Heavy>,
    warm: Mutex<Option<ResidentPair<Text, Heavy>>>,
    on_staged: Option<Box<dyn Fn() + Send + Sync>>,
}

impl<Text, Heavy> Residency<Text, Heavy> {
    /// The warm-cache policy: hold every component resident. `text` and `heavy` are built once at
    /// load; [`run`](Self::run) borrows them for every generation.
    pub fn resident(text: Text, heavy: Heavy) -> Self {
        Self {
            // Compatibility constructor for in-memory providers that do not expose request-scoped
            // staging. The shared shape still retains two loaders, but calling either is a typed
            // error rather than fabricating a reload source that does not exist.
            loaders: SeqLoaders {
                load_text: Box::new(|_| {
                    Err(Error::Msg(
                        "resident-only component source cannot reload text".into(),
                    ))
                }),
                load_heavy: Box::new(|_, _| {
                    Err(Error::Msg(
                        "resident-only component source cannot reload heavy components".into(),
                    ))
                }),
            },
            warm: Mutex::new(Some(ResidentPair {
                text,
                heavy,
                streamable: false,
            })),
            on_staged: None,
        }
    }

    /// The peak-bounding policy: hold only the two per-phase loader closures. `load_text` rebuilds the
    /// phase-A component; `load_heavy(use_pid)` rebuilds the render bundle, skipping the PiD overlay
    /// when `use_pid` is `false` (F-177). Both are re-run on every [`run`](Self::run) and their
    /// products freed afterward, so nothing stays resident across jobs.
    ///
    /// The closures must produce components **byte-identical** to the `Resident` path's — the
    /// A/B parity that the `sequential_residency_real_weights` suites assert rests on it.
    pub fn sequential(
        load_text: impl Fn() -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: SeqLoaders {
                load_text: Box::new(move |_| load_text()),
                load_heavy: Box::new(move |use_pid, _| load_heavy(use_pid)),
            },
            warm: Mutex::new(None),
            on_staged: None,
        }
    }

    /// Build a request-scoped residency. Construction retains only loaders; the first request
    /// decides whether to populate the warm pair or run the phase-staged lifecycle.
    pub fn request_scoped(
        load_text: impl Fn(bool) -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool, bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: SeqLoaders {
                load_text: Box::new(load_text),
                load_heavy: Box::new(load_heavy),
            },
            warm: Mutex::new(None),
            on_staged: None,
        }
    }

    /// Attach a once-per-provider advisory hook for staged execution.
    pub fn with_staged_advisory(mut self, advisory: impl Fn() + Send + Sync + 'static) -> Self {
        self.on_staged = Some(Box::new(advisory));
        self
    }

    /// Whether no warm pair is currently cached. Legacy policy-based providers use this as their
    /// resident/sequential discriminator; request-scoped providers use it only for observation.
    pub fn is_sequential(&self) -> bool {
        self.warm
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_none()
    }

    /// Run `f` while borrowing the warm pair, or return `None` when no pair is cached.
    pub fn with_resident_parts<R>(&self, f: impl FnOnce(&Text, &Heavy) -> R) -> Option<R> {
        let warm = self
            .warm
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        warm.as_ref().map(|pair| f(&pair.text, &pair.heavy))
    }

    /// The single dispatch every wired provider shares (sc-11126, F-180): map an
    /// [`OffloadPolicy`] to a [`Residency`] built from the two per-phase loader closures, so no
    /// provider re-derives the `match policy { … }` (the earlier per-crate copies is exactly what let
    /// a sibling silently ignore `offload_policy`). `load_text` / `load_heavy` are the same closures
    /// [`sequential`](Self::sequential) captures:
    ///
    /// * [`OffloadPolicy::Resident`] runs **both eagerly now** and holds the products warm — the heavy
    ///   loader is invoked with `use_pid = true` so any PiD overlay in the spec is loaded once and
    ///   reused across generates (the per-request F-177 skip applies only to the `Sequential`
    ///   per-generate loader, which re-runs `load_heavy` each generate).
    /// * [`OffloadPolicy::Sequential`] captures the closures and loads **nothing now**; each
    ///   [`run`](Self::run) re-runs them in phase order and frees the products.
    ///
    /// The deferral is the discriminator a weight-free test asserts: under `Sequential` this returns
    /// without touching either loader, so a dispatch that mapped `Sequential → Resident` (the
    /// ignore-`offload_policy` bug) would eager-load here and fail the "no load at construction" check.
    pub fn from_policy(
        policy: OffloadPolicy,
        load_text: impl Fn() -> Result<Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> Result<Heavy> + Send + Sync + 'static,
    ) -> Result<Self> {
        let loaders = SeqLoaders {
            load_text: Box::new(move |_| load_text()),
            load_heavy: Box::new(move |use_pid, _| load_heavy(use_pid)),
        };
        match policy {
            OffloadPolicy::Resident => {
                let text = (loaders.load_text)(false)?;
                let heavy = (loaders.load_heavy)(true, false)?;
                Ok(Self {
                    loaders,
                    warm: Mutex::new(Some(ResidentPair {
                        text,
                        heavy,
                        streamable: false,
                    })),
                    on_staged: None,
                })
            }
            OffloadPolicy::Sequential => Ok(Self {
                loaders,
                warm: Mutex::new(None),
                on_staged: None,
            }),
        }
    }

    fn warm(&self) -> Result<MutexGuard<'_, Option<ResidentPair<Text, Heavy>>>> {
        self.warm
            .lock()
            .map_err(|_| Error::Msg("component-residency mutex poisoned".into()))
    }

    /// Drop the warm pair and flush MLX's allocator before any replacement load starts.
    fn evict_warm_locked(warm: &mut Option<ResidentPair<Text, Heavy>>) -> bool {
        let Some(pair) = warm.take() else {
            return false;
        };
        drop(pair);
        note_clear_cache();
        mlx_rs::memory::clear_cache();
        true
    }

    /// Ensure a warm pair exists in the requested component form. A form change is an eviction, and
    /// eviction completes (drop + cache flush) before either loader runs.
    fn ensure_warm_locked(
        &self,
        warm: &mut Option<ResidentPair<Text, Heavy>>,
        streamable: bool,
        _use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<()> {
        let mismatch = warm
            .as_ref()
            .is_some_and(|pair| pair.streamable != streamable);
        if mismatch {
            Self::evict_warm_locked(warm);
        }
        if warm.is_none() {
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            let text = (self.loaders.load_text)(streamable)?;
            on_progress(Progress::Loading(LoadPhase::Renderer));
            // A warm pair is reused across requests, so it must include any configured optional
            // overlay even when the request that populates it does not use that overlay. This
            // preserves the legacy Resident guarantee and prevents a later use_pid=true request
            // from seeing a warm bundle built for use_pid=false.
            let heavy = (self.loaders.load_heavy)(true, streamable)?;
            *warm = Some(ResidentPair {
                text,
                heavy,
                streamable,
            });
        }
        Ok(())
    }

    /// Drive one generation through the staged residency lifecycle, running identically for both
    /// policies:
    ///
    /// 1. cancel check (before any work),
    /// 2. `encode` the prompt with the phase-A `Text` component,
    /// 3. under `Sequential` only: `materialize` the encode outputs (force MLX to evaluate them while
    ///    the encoder is still alive — MLX is lazy, so an un-evaluated output keeps the encoder
    ///    weights referenced through the graph and freeing it would reclaim nothing), then free the
    ///    encoder + `clear_cache()`,
    /// 4. cancel check (before the heavy load),
    /// 5. load the `Heavy` bundle (threading `use_pid`),
    /// 6. cancel check (after the heavy load),
    /// 7. `render` from the `Heavy` bundle + the encode outputs,
    /// 8. under `Sequential` only: free the heavy bundle + `clear_cache()`.
    ///
    /// Steps 3 and 8 run under `Sequential` only, and their `clear_cache()` runs on **every** exit —
    /// success, error, or cancel — via an RAII guard (F-174). `materialize` is never called under
    /// `Resident` (which does not eval), preserving byte-identical warm-cache behavior.
    ///
    /// `on_progress` receives a [`Progress::Loading`] event before each `Sequential` in-`generate`
    /// component load (F-179) — the multi-GB text-encoder and heavy-bundle loads that fire *inside*
    /// `generate`, during which no `Step`/`Decoding` event otherwise reaches the UI. It is emitted only
    /// under `Sequential` (the `Resident` path loads before `generate`), and is threaded on to `render`
    /// so the denoise/decode body keeps emitting `Step`/`Decoding` as before.
    ///
    /// `E` is the provider's encode output (conditioning tensors); `Out` is the generation result.
    pub fn run<E, Out>(
        &self,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> Result<E>,
        materialize: impl FnOnce(&E) -> Result<()>,
        render: impl FnOnce(&Heavy, E, &mut dyn FnMut(Progress)) -> Result<Out>,
    ) -> Result<Out> {
        let stage_residency = self.is_sequential();
        self.run_request_scoped(
            stage_residency,
            false,
            cancel,
            use_pid,
            on_progress,
            encode,
            materialize,
            render,
        )
    }

    /// Request-scoped counterpart to [`run`](Self::run). `stage_residency` chooses the lifecycle for
    /// this call; `streamable` chooses the loader form independently.
    #[allow(clippy::too_many_arguments)]
    pub fn run_request_scoped<E, Out>(
        &self,
        stage_residency: bool,
        streamable: bool,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> Result<E>,
        materialize: impl FnOnce(&E) -> Result<()>,
        render: impl FnOnce(&Heavy, E, &mut dyn FnMut(Progress)) -> Result<Out>,
    ) -> Result<Out> {
        // F-173: a request cancelled before the (Sequential) load preamble returns promptly.
        check_cancel(cancel)?;
        // The warm-state mutex is also the request lifecycle lock. Holding it through either path
        // prevents a concurrent warm request from repopulating the cache while a staged request is
        // loading its own components.
        let mut warm = self.warm()?;
        if !stage_residency {
            self.ensure_warm_locked(&mut warm, streamable, use_pid, on_progress)?;
            let pair = warm
                .as_ref()
                .ok_or_else(|| Error::Msg("warm residency was not populated".into()))?;
            let enc = encode(&pair.text)?;
            check_cancel(cancel)?;
            let result = render(&pair.heavy, enc, on_progress);
            drop(warm);
            return result;
        }

        if let Some(advisory) = &self.on_staged {
            advisory();
        }
        // Eviction completes before the first staged loader starts. This is the ordering that keeps a
        // warm pair and its replacement from overlapping at the request peak.
        Self::evict_warm_locked(&mut warm);
        let enc = {
            let _text_cleanup = ClearCacheGuard;
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            let text = (self.loaders.load_text)(streamable)?;
            let enc = encode(&text)?;
            materialize(&enc)?;
            enc
        };
        check_cancel(cancel)?;
        let _heavy_cleanup = ClearCacheGuard;
        on_progress(Progress::Loading(LoadPhase::Renderer));
        let heavy = (self.loaders.load_heavy)(use_pid, streamable)?;
        check_cancel(cancel)?;
        let result = render(&heavy, enc, on_progress);
        drop(warm);
        result
    }
}

impl<Text, Heavy: StagedHeavy> Residency<Text, Heavy> {
    /// Like [`run`](Self::run), but frees the DiT-bearing heavy bundle **before** the decode phase
    /// under `Sequential` (sc-13571, GitHub #1658) — bounding the decode-phase peak to the light (VAE)
    /// bundle plus the tiled-decode transient, rather than that sum PLUS the still-resident DiT
    /// (~3.2 GiB for z-image q4). The phases:
    ///
    /// 1. `encode` the prompt from the phase-A text component (freed after, under `Sequential`),
    /// 2. `denoise` from `&Heavy` into intermediate latents `Mid`,
    /// 3. under `Sequential` only: `materialize_mid` (force MLX to evaluate the latents while the DiT is
    ///    still alive — MLX is lazy, so an un-evaluated `Mid` keeps the DiT referenced through the graph
    ///    and the shed would free nothing), then [`shed_dit`](StagedHeavy::shed_dit) + `clear_cache()`,
    /// 4. `decode` the latents through the light decode view.
    ///
    /// Under `Resident` nothing is shed (the bundle stays warm across jobs): step 3 is skipped and
    /// `decode` runs against the still-held bundle's [`decode_view`](StagedHeavy::decode_view). The
    /// `clear_cache()` discipline mirrors [`run`](Self::run): `Sequential` clears on every exit (via the
    /// RAII guards, so a mid-render `?` early return still flushes), `Resident` never does.
    #[allow(clippy::too_many_arguments)]
    pub fn run_staged<E, Mid, Out>(
        &self,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> Result<E>,
        materialize_enc: impl FnOnce(&E) -> Result<()>,
        denoise: impl FnOnce(&Heavy, E, &mut dyn FnMut(Progress)) -> Result<Mid>,
        materialize_mid: impl FnOnce(&Mid) -> Result<()>,
        decode: impl FnOnce(Heavy::DecodeView<'_>, Mid, &mut dyn FnMut(Progress)) -> Result<Out>,
    ) -> Result<Out> {
        let stage_residency = self.is_sequential();
        self.run_staged_request_scoped(
            stage_residency,
            false,
            cancel,
            use_pid,
            on_progress,
            encode,
            materialize_enc,
            denoise,
            materialize_mid,
            decode,
        )
    }

    /// Request-scoped counterpart to [`run_staged`](Self::run_staged).
    #[allow(clippy::too_many_arguments)]
    pub fn run_staged_request_scoped<E, Mid, Out>(
        &self,
        stage_residency: bool,
        streamable: bool,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> Result<E>,
        materialize_enc: impl FnOnce(&E) -> Result<()>,
        denoise: impl FnOnce(&Heavy, E, &mut dyn FnMut(Progress)) -> Result<Mid>,
        materialize_mid: impl FnOnce(&Mid) -> Result<()>,
        decode: impl FnOnce(Heavy::DecodeView<'_>, Mid, &mut dyn FnMut(Progress)) -> Result<Out>,
    ) -> Result<Out> {
        check_cancel(cancel)?;
        let mut warm = self.warm()?;
        if !stage_residency {
            self.ensure_warm_locked(&mut warm, streamable, use_pid, on_progress)?;
            let pair = warm
                .as_ref()
                .ok_or_else(|| Error::Msg("warm residency was not populated".into()))?;
            let enc = encode(&pair.text)?;
            check_cancel(cancel)?;
            let mid = denoise(&pair.heavy, enc, on_progress)?;
            let result = decode(pair.heavy.decode_view(), mid, on_progress);
            drop(warm);
            return result;
        }

        if let Some(advisory) = &self.on_staged {
            advisory();
        }
        Self::evict_warm_locked(&mut warm);
        let enc = {
            let _text_cleanup = ClearCacheGuard;
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            let text = (self.loaders.load_text)(streamable)?;
            let enc = encode(&text)?;
            materialize_enc(&enc)?;
            enc
        };
        check_cancel(cancel)?;
        let (light, mid) = {
            // The guard covers the heavy load itself and every fallible operation through the DiT
            // shed. On success it flushes immediately after the DiT drops and before decode; on
            // error/cancel it flushes after the partially/fully loaded heavy bundle drops.
            let _heavy_cleanup = ClearCacheGuard;
            on_progress(Progress::Loading(LoadPhase::Renderer));
            let heavy = (self.loaders.load_heavy)(use_pid, streamable)?;
            check_cancel(cancel)?;
            let mid = denoise(&heavy, enc, on_progress)?;
            materialize_mid(&mid)?;
            (heavy.shed_dit(), mid)
        };
        let result = {
            // Guard-before-value ordering makes the light bundle drop before its final cache flush.
            let _light_cleanup = ClearCacheGuard;
            let light = light;
            decode(Heavy::light_view(&light), mid, on_progress)
        };
        drop(warm);
        result
    }
}

/// Map a tripped [`CancelFlag`] to the typed [`Error::Canceled`] (which bridges 1:1 to
/// `gen_core::Error::Canceled`; never laundered through `Error::backend`).
fn check_cancel(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}

/// Emit the F-181 advisory: staged/`Sequential` execution over a **dense** quantized snapshot re-quantizes
/// the whole model on every generate (repeated compute) and the dense transient means the per-phase
/// peak is the *dense* component size, shrinking the memory win. Packed (pre-quantized) snapshots
/// avoid both. Providers call this from their `Sequential` load arm when they detect that combination;
/// the workspace has no `log` crate, so it goes to stderr like the other load-time advisories
/// (e.g. SDXL's `SDXL_LORA_VENDORED`).
pub fn warn_sequential_requantize(model_id: &str, bits: i32) {
    eprintln!(
        "{model_id}: staged residency with Q{bits} over a dense snapshot re-quantizes the whole \
         model on EVERY generate (repeated compute; the dense transient makes the per-phase peak the \
         dense component size, not the packed one — shrinking the memory win). Point at a \
         pre-quantized Q{bits} snapshot to avoid both."
    );
}

/// RAII guard that calls [`mlx_rs::memory::clear_cache`] on drop — so a `Sequential` generate returns
/// MLX's buffer-cache pages to the OS on **every** exit (success, error, cancel), not only the success
/// tail the pre-seam copies covered (F-174). Declared before the component it guards so the component
/// drops first (freeing the working set) and the cache flush fires after.
struct ClearCacheGuard;

impl Drop for ClearCacheGuard {
    fn drop(&mut self) {
        note_clear_cache();
        mlx_rs::memory::clear_cache();
    }
}

// Test-only observation hook: records that a `ClearCacheGuard` fired, so the weight-free state-
// machine tests can assert the cache flush runs on each exit path without inspecting MLX internals.
// A no-op outside tests.
#[cfg(test)]
thread_local! {
    static CLEAR_CACHE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn note_clear_cache() {
    #[cfg(test)]
    CLEAR_CACHE_CALLS.with(|c| c.set(c.get() + 1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// A shared event log the fake components/closures append to, so a test can assert the exact
    /// order of load/encode/materialize/render/drop across the staged lifecycle. `Arc<Mutex>` (not
    /// `Rc`) so the fakes satisfy the seam's `Send + Sync` loader bound.
    type Log = Arc<Mutex<Vec<String>>>;

    fn new_log() -> Log {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn record(l: &Log, msg: &str) {
        l.lock().unwrap().push(msg.to_string());
    }

    fn events(l: &Log) -> Vec<String> {
        l.lock().unwrap().clone()
    }

    /// A fake phase-A encoder that records its drop — proving the `Sequential` path frees it (F-174).
    struct FakeText {
        log: Log,
    }
    impl Drop for FakeText {
        fn drop(&mut self) {
            record(&self.log, "drop_text");
        }
    }

    /// A fake heavy bundle recording whether it was built with PiD (F-177) and recording its drop.
    struct FakeHeavy {
        log: Log,
        with_pid: bool,
    }
    impl Drop for FakeHeavy {
        fn drop(&mut self) {
            record(
                &self.log,
                if self.with_pid {
                    "drop_heavy_pid"
                } else {
                    "drop_heavy_nopid"
                },
            );
        }
    }

    /// The fake light (decode-only) bundle that survives the DiT drop — records its own drop so a test
    /// can assert the decode runs against it (sc-13571).
    struct FakeLight {
        log: Log,
    }
    impl Drop for FakeLight {
        fn drop(&mut self) {
            record(&self.log, "drop_light");
        }
    }
    /// A borrowed decode view over the shared log — produced from either the warm heavy (`Resident`) or
    /// the shed light (`Sequential`), so the decode closure records identically for both.
    struct FakeDecodeView<'a> {
        log: &'a Log,
    }
    impl StagedHeavy for FakeHeavy {
        type Light = FakeLight;
        type DecodeView<'a> = FakeDecodeView<'a>;
        fn shed_dit(self) -> FakeLight {
            // `self` (the DiT-bearing heavy) drops here → records `drop_heavy_*`, the DiT freed.
            FakeLight {
                log: self.log.clone(),
            }
        }
        fn decode_view(&self) -> FakeDecodeView<'_> {
            FakeDecodeView { log: &self.log }
        }
        fn light_view(light: &FakeLight) -> FakeDecodeView<'_> {
            FakeDecodeView { log: &light.log }
        }
    }

    fn clear_cache_calls() -> usize {
        CLEAR_CACHE_CALLS.with(|c| c.get())
    }
    fn reset_clear_cache_calls() {
        CLEAR_CACHE_CALLS.with(|c| c.set(0));
    }

    /// A progress sink that ignores every event — used where a test asserts loads/drops, not progress.
    fn ignore_progress(_p: Progress) {}

    fn seq_residency(log: &Log) -> Residency<FakeText, FakeHeavy> {
        let lt = log.clone();
        let lh = log.clone();
        Residency::sequential(
            move || {
                record(&lt, "load_text");
                Ok(FakeText { log: lt.clone() })
            },
            move |use_pid| {
                record(
                    &lh,
                    if use_pid {
                        "load_heavy_pid"
                    } else {
                        "load_heavy_nopid"
                    },
                );
                Ok(FakeHeavy {
                    log: lh.clone(),
                    with_pid: use_pid,
                })
            },
        )
    }

    fn request_residency(log: &Log) -> Residency<FakeText, FakeHeavy> {
        let lt = log.clone();
        let lh = log.clone();
        Residency::request_scoped(
            move |streamable| {
                record(&lt, &format!("load_text_streamable_{streamable}"));
                Ok(FakeText { log: lt.clone() })
            },
            move |use_pid, streamable| {
                record(
                    &lh,
                    &format!("load_heavy_pid_{use_pid}_streamable_{streamable}"),
                );
                Ok(FakeHeavy {
                    log: lh.clone(),
                    with_pid: use_pid,
                })
            },
        )
    }

    fn request_run(
        residency: &Residency<FakeText, FakeHeavy>,
        stage_residency: bool,
        streamable: bool,
        log: &Log,
    ) -> u32 {
        let encode_log = log.clone();
        let materialize_log = log.clone();
        let render_log = log.clone();
        residency
            .run_request_scoped(
                stage_residency,
                streamable,
                &CancelFlag::new(),
                false,
                &mut ignore_progress,
                move |_text| {
                    record(&encode_log, "encode");
                    Ok(41u32)
                },
                move |_encoded| {
                    record(&materialize_log, "materialize");
                    Ok(())
                },
                move |_heavy, encoded, _progress| {
                    record(&render_log, "render");
                    Ok(encoded + 1)
                },
            )
            .unwrap()
    }

    #[test]
    fn request_scoped_construction_loads_nothing_and_warm_reuses_one_pair() {
        let log = new_log();
        let residency = request_residency(&log);
        assert!(
            events(&log).is_empty(),
            "construction must retain only loaders"
        );

        assert_eq!(request_run(&residency, false, false, &log), 42);
        assert_eq!(request_run(&residency, false, false, &log), 42);
        let events = events(&log);
        assert_eq!(
            events
                .iter()
                .filter(|event| *event == "load_text_streamable_false")
                .count(),
            1,
            "two warm requests must reuse one text component"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| *event == "load_heavy_pid_true_streamable_false")
                .count(),
            1,
            "two warm requests must reuse one heavy component"
        );
        assert!(
            !events.iter().any(|event| event == "materialize"),
            "warm execution must preserve the non-materializing path"
        );
    }

    #[test]
    fn warm_staged_warm_evicts_before_loading_and_rebuilds_after_staging() {
        let log = new_log();
        let residency = request_residency(&log);
        reset_clear_cache_calls();

        assert_eq!(request_run(&residency, false, false, &log), 42);
        assert_eq!(request_run(&residency, true, false, &log), 42);
        assert_eq!(request_run(&residency, false, false, &log), 42);

        assert_eq!(
            events(&log),
            [
                "load_text_streamable_false",
                "load_heavy_pid_true_streamable_false",
                "encode",
                "render",
                "drop_text",
                "drop_heavy_pid",
                "load_text_streamable_false",
                "encode",
                "materialize",
                "drop_text",
                "load_heavy_pid_false_streamable_false",
                "render",
                "drop_heavy_nopid",
                "load_text_streamable_false",
                "load_heavy_pid_true_streamable_false",
                "encode",
                "render",
            ],
            "warm pair must drop before staged loading; staged leaves no pair for the next warm request"
        );
        assert_eq!(
            clear_cache_calls(),
            3,
            "warm eviction plus staged text/heavy cleanup each flush once"
        );
    }

    #[test]
    fn changing_component_form_evicts_before_reloading() {
        let log = new_log();
        let residency = request_residency(&log);
        assert_eq!(request_run(&residency, false, false, &log), 42);
        assert_eq!(request_run(&residency, false, true, &log), 42);
        assert_eq!(
            events(&log),
            [
                "load_text_streamable_false",
                "load_heavy_pid_true_streamable_false",
                "encode",
                "render",
                "drop_text",
                "drop_heavy_pid",
                "load_text_streamable_true",
                "load_heavy_pid_true_streamable_true",
                "encode",
                "render",
            ]
        );
    }

    #[test]
    fn concurrent_warm_and_staged_requests_never_overlap_component_loads() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let log = new_log();
        let timed_load = |active: Arc<AtomicUsize>, max_active: Arc<AtomicUsize>| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            active.fetch_sub(1, Ordering::SeqCst);
        };
        let residency = Arc::new(Residency::request_scoped(
            {
                let active = active.clone();
                let max_active = max_active.clone();
                let log = log.clone();
                move |_| {
                    timed_load(active.clone(), max_active.clone());
                    Ok(FakeText { log: log.clone() })
                }
            },
            {
                let active = active.clone();
                let max_active = max_active.clone();
                let log = log.clone();
                move |use_pid, _| {
                    timed_load(active.clone(), max_active.clone());
                    Ok(FakeHeavy {
                        log: log.clone(),
                        with_pid: use_pid,
                    })
                }
            },
        ));
        let start = Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            for stage in [false, true] {
                let residency = residency.clone();
                let start = start.clone();
                scope.spawn(move || {
                    start.wait();
                    residency
                        .run_request_scoped(
                            stage,
                            false,
                            &CancelFlag::new(),
                            false,
                            &mut ignore_progress,
                            |_| Ok(()),
                            |_| Ok(()),
                            |_, _, _| Ok(()),
                        )
                        .unwrap();
                });
            }
            start.wait();
        });
        assert_eq!(
            max_active.load(Ordering::SeqCst),
            1,
            "warm and staged requests loaded components concurrently"
        );
    }

    #[test]
    fn run_staged_sequential_sheds_the_dit_before_the_decode() {
        // sc-13571 / GitHub #1658: the DiT (heavy bundle) must be dropped + clear_cache'd AFTER denoise
        // and BEFORE the decode, so the decode-phase peak excludes it.
        let log: Log = new_log();
        let res = seq_residency(&log);
        reset_clear_cache_calls();
        let l = log.clone();
        res.run_staged(
            &CancelFlag::new(),
            false,
            &mut ignore_progress,
            |_text| {
                record(&l, "encode");
                Ok(())
            },
            |()| {
                record(&l, "materialize_enc");
                Ok(())
            },
            |_heavy, (), _op| {
                record(&l, "denoise");
                Ok(())
            },
            |()| {
                record(&l, "materialize_mid");
                Ok(())
            },
            |view: FakeDecodeView, (), _op| {
                record(view.log, "decode");
                Ok(())
            },
        )
        .unwrap();
        let ev = events(&log);
        assert_eq!(
            ev,
            [
                "load_text",
                "encode",
                "materialize_enc",
                "drop_text",
                "load_heavy_nopid",
                "denoise",
                "materialize_mid",
                "drop_heavy_nopid", // the DiT is shed …
                "decode",           // … BEFORE the decode runs
                "drop_light",
            ],
            "staged Sequential order"
        );
        // clear_cache fires for the text drop, the DiT shed, and the light drop.
        assert_eq!(clear_cache_calls(), 3);
    }

    #[test]
    fn run_staged_denoise_error_drops_heavy_and_flushes_before_returning() {
        let log = new_log();
        let residency = seq_residency(&log);
        reset_clear_cache_calls();
        let error = residency
            .run_staged_request_scoped(
                true,
                false,
                &CancelFlag::new(),
                false,
                &mut ignore_progress,
                |_| Ok(()),
                |_| Ok(()),
                |_, (), _| Err::<(), _>(Error::Msg("denoise failed".into())),
                |_| Ok(()),
                |_, (), _| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(error, Error::Msg(_)));
        assert_eq!(
            events(&log),
            [
                "load_text",
                "drop_text",
                "load_heavy_nopid",
                "drop_heavy_nopid",
            ]
        );
        assert_eq!(
            clear_cache_calls(),
            2,
            "text and heavy error paths must each flush"
        );
    }

    #[test]
    fn run_staged_resident_keeps_the_dit_warm_and_never_clears() {
        // Under `Resident` nothing is shed (the bundle stays warm across jobs) and no materialize /
        // clear_cache runs — the decode borrows the still-held bundle's `decode_view`.
        let log: Log = new_log();
        let res = Residency::resident(
            FakeText { log: log.clone() },
            FakeHeavy {
                log: log.clone(),
                with_pid: true,
            },
        );
        reset_clear_cache_calls();
        let l = log.clone();
        res.run_staged(
            &CancelFlag::new(),
            false,
            &mut ignore_progress,
            |_text| {
                record(&l, "encode");
                Ok(())
            },
            |()| {
                record(&l, "materialize_enc");
                Ok(())
            },
            |_heavy, (), _op| {
                record(&l, "denoise");
                Ok(())
            },
            |()| {
                record(&l, "materialize_mid");
                Ok(())
            },
            |view: FakeDecodeView, (), _op| {
                record(view.log, "decode");
                Ok(())
            },
        )
        .unwrap();
        // No materialize, no shed, no drops during the run; the heavy is still held (drops only when
        // `res` does, after this snapshot).
        assert_eq!(events(&log), ["encode", "denoise", "decode"]);
        assert_eq!(clear_cache_calls(), 0);
    }

    #[test]
    fn resident_runs_stages_in_order_without_eval_or_clear() {
        let log: Log = new_log();
        let res = Residency::resident(
            FakeText { log: log.clone() },
            FakeHeavy {
                log: log.clone(),
                with_pid: true,
            },
        );
        reset_clear_cache_calls();
        let l = log.clone();
        let out = res
            .run(
                &CancelFlag::new(),
                false,
                &mut ignore_progress,
                |_t| {
                    record(&l, "encode");
                    Ok(7u32)
                },
                |_e| {
                    // Resident must NEVER materialize (byte-identical warm path).
                    record(&l, "materialize");
                    Ok(())
                },
                |_h, e, _op| {
                    record(&l, "render");
                    Ok(e + 1)
                },
            )
            .unwrap();
        assert_eq!(out, 8);
        assert_eq!(events(&log), vec!["encode", "render"]);
        // Resident holds its components — no per-generate clear_cache, no eval.
        assert_eq!(clear_cache_calls(), 0);
    }

    #[test]
    fn sequential_runs_full_lifecycle_in_order() {
        let log: Log = new_log();
        let res = seq_residency(&log);
        reset_clear_cache_calls();
        let l = log.clone();
        // Record the progress events too, so this test also pins the F-179 Loading signals.
        let prog = new_log();
        let pl = prog.clone();
        let mut on_progress = |p: Progress| {
            if let Progress::Loading(phase) = p {
                record(&pl, &format!("loading:{phase:?}"));
            }
        };
        let out = res
            .run(
                &CancelFlag::new(),
                true,
                &mut on_progress,
                |_t| {
                    record(&l, "encode");
                    Ok(1u32)
                },
                |_e| {
                    record(&l, "materialize");
                    Ok(())
                },
                |_h, e, _op| {
                    record(&l, "render");
                    Ok(e)
                },
            )
            .unwrap();
        assert_eq!(out, 1);
        assert_eq!(
            events(&log),
            vec![
                "load_text",
                "encode",
                "materialize",
                "drop_text",
                "load_heavy_pid",
                "render",
                "drop_heavy_pid",
            ]
        );
        // clear_cache fires twice: once after the text drop, once after the heavy drop.
        assert_eq!(clear_cache_calls(), 2);
        // F-179: a Loading signal fires before EACH Sequential component load, in phase order.
        assert_eq!(
            events(&prog),
            vec!["loading:TextEncoder", "loading:Renderer"]
        );
    }

    #[test]
    fn sequential_skips_pid_load_when_not_used() {
        // F-177: use_pid=false ⇒ the heavy loader is invoked with false ⇒ no PiD student is built.
        let log: Log = new_log();
        let res = seq_residency(&log);
        res.run(
            &CancelFlag::new(),
            false,
            &mut ignore_progress,
            |_t| Ok(()),
            |_e| Ok(()),
            |_h, _e, _op| Ok(()),
        )
        .unwrap();
        let ev = events(&log);
        assert!(ev.iter().any(|e| e == "load_heavy_nopid"), "{ev:?}");
        assert!(!ev.iter().any(|e| e == "load_heavy_pid"), "{ev:?}");
        assert!(ev.iter().any(|e| e == "drop_heavy_nopid"), "{ev:?}");
    }

    #[test]
    fn cancel_before_encode_returns_promptly() {
        // F-173: an already-cancelled request does no load/encode work at all.
        let log: Log = new_log();
        let res = seq_residency(&log);
        let cancel = CancelFlag::new();
        cancel.cancel();
        let err = res
            .run(
                &cancel,
                true,
                &mut ignore_progress,
                |_t| Ok(()),
                |_e| Ok(()),
                |_h, _e, _op| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Canceled));
        assert!(
            events(&log).is_empty(),
            "no stage should run: {:?}",
            events(&log)
        );
    }

    #[test]
    fn cancel_between_encode_and_heavy_frees_text_and_skips_heavy() {
        // F-173 + F-174: a cancel raised during encode trips the pre-heavy boundary — the heavy bundle
        // is never loaded, but the text encoder was freed (drop_text) and its cache flushed.
        let log: Log = new_log();
        let res = seq_residency(&log);
        reset_clear_cache_calls();
        let cancel = CancelFlag::new();
        let c2 = cancel.clone();
        let err = res
            .run(
                &cancel,
                true,
                &mut ignore_progress,
                |_t| {
                    c2.cancel(); // simulate a cancel arriving mid-encode
                    Ok(())
                },
                |_e| Ok(()),
                |_h: &FakeHeavy, _e: (), _op: &mut dyn FnMut(Progress)| -> Result<()> {
                    panic!("render must not run after a pre-heavy cancel")
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::Canceled));
        let ev = events(&log);
        assert_eq!(
            ev,
            vec!["load_text", "drop_text"],
            "text loaded+freed, heavy never loaded"
        );
        // The Phase-A guard still flushed the cache on the cancel path.
        assert_eq!(clear_cache_calls(), 1);
    }

    #[test]
    fn cancel_after_heavy_load_frees_heavy_and_skips_render() {
        // F-173 + F-174: a cancel that lands DURING the heavy load must trip the post-load boundary —
        // render is skipped, but the heavy bundle is freed (drop_heavy) with its cache flushed.
        let log: Log = new_log();
        let cancel = CancelFlag::new();
        let cflag = cancel.clone();
        let lt = log.clone();
        let lh = log.clone();
        let res = Residency::sequential(
            move || {
                record(&lt, "load_text");
                Ok(FakeText { log: lt.clone() })
            },
            move |use_pid| {
                record(&lh, "load_heavy");
                cflag.cancel(); // cancel arrives during the heavy load
                Ok(FakeHeavy {
                    log: lh.clone(),
                    with_pid: use_pid,
                })
            },
        );
        reset_clear_cache_calls();
        let err = res
            .run(
                &cancel,
                true,
                &mut ignore_progress,
                |_t| Ok(()),
                |_e| Ok(()),
                |_h: &FakeHeavy, _e: (), _op: &mut dyn FnMut(Progress)| -> Result<()> {
                    panic!("render must not run after a post-heavy-load cancel")
                },
            )
            .unwrap_err();
        assert!(matches!(err, Error::Canceled));
        let ev = events(&log);
        assert_eq!(
            ev,
            vec!["load_text", "drop_text", "load_heavy", "drop_heavy_pid"],
            "heavy loaded then freed; render skipped"
        );
        // Both guards flushed: once after text, once after heavy.
        assert_eq!(clear_cache_calls(), 2);
    }

    #[test]
    fn error_in_render_still_frees_heavy() {
        // F-174: a render error frees the heavy bundle and flushes the cache (the leak the copies had).
        let log: Log = new_log();
        let res = seq_residency(&log);
        reset_clear_cache_calls();
        let err = res
            .run(
                &CancelFlag::new(),
                true,
                &mut ignore_progress,
                |_t| Ok(()),
                |_e| Ok(()),
                |_h, _e: (), _op: &mut dyn FnMut(Progress)| Err::<(), _>(Error::Msg("boom".into())),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Msg(_)));
        let ev = events(&log);
        assert!(ev.iter().any(|e| e == "drop_heavy_pid"), "{ev:?}");
        assert_eq!(clear_cache_calls(), 2);
    }

    #[test]
    fn resident_emits_no_loading_progress() {
        // F-179: the Resident path loads before `generate`, so it never emits a Loading event — the
        // signal is Sequential-only. (The Sequential ordering is asserted in
        // `sequential_runs_full_lifecycle_in_order`.)
        let log: Log = new_log();
        let res = Residency::resident(
            FakeText { log: log.clone() },
            FakeHeavy {
                log: log.clone(),
                with_pid: true,
            },
        );
        let mut loading = 0usize;
        res.run(
            &CancelFlag::new(),
            false,
            &mut |p| {
                if matches!(p, Progress::Loading(_)) {
                    loading += 1;
                }
            },
            |_t| Ok(0u32),
            |_e| Ok(()),
            |_h, e, _op| Ok(e),
        )
        .unwrap();
        assert_eq!(loading, 0, "Resident must not emit Loading");
    }

    // ── F-180: the shared `from_policy` dispatch every wired provider routes through. These are the
    // meaningful (not smoke) state-machine assertions: `Sequential` must DEFER — construct without
    // touching either loader — so a dispatch that mapped `Sequential → Resident` (the ignore-
    // `offload_policy` bug the 2026-07-11 review flagged) would eager-load here and fail the count.

    /// A loader pair that counts how many times each is invoked, so a test can prove `Sequential`
    /// defers (0/0 at construction) while `Resident` loads eagerly (1/1).
    fn counting_loaders(
        text_calls: &Arc<Mutex<usize>>,
        heavy_calls: &Arc<Mutex<usize>>,
    ) -> (
        impl Fn() -> Result<FakeText> + Send + Sync + 'static,
        impl Fn(bool) -> Result<FakeHeavy> + Send + Sync + 'static,
    ) {
        let (tc, hc) = (text_calls.clone(), heavy_calls.clone());
        let log = new_log();
        let (lt, lh) = (log.clone(), log);
        (
            move || {
                *tc.lock().unwrap() += 1;
                Ok(FakeText { log: lt.clone() })
            },
            move |use_pid| {
                *hc.lock().unwrap() += 1;
                Ok(FakeHeavy {
                    log: lh.clone(),
                    with_pid: use_pid,
                })
            },
        )
    }

    #[test]
    fn from_policy_sequential_defers_all_loads() {
        let text_calls = Arc::new(Mutex::new(0usize));
        let heavy_calls = Arc::new(Mutex::new(0usize));
        let (lt, lh) = counting_loaders(&text_calls, &heavy_calls);
        let res = Residency::from_policy(OffloadPolicy::Sequential, lt, lh).unwrap();
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential residency"
        );
        // The discriminator: neither loader ran at construction. A dispatch that ignored the policy
        // and always went Resident would have eager-loaded, tripping these to 1.
        assert_eq!(
            *text_calls.lock().unwrap(),
            0,
            "Sequential must not load the text encoder eagerly"
        );
        assert_eq!(
            *heavy_calls.lock().unwrap(),
            0,
            "Sequential must not load the heavy bundle eagerly"
        );
    }

    #[test]
    fn from_policy_resident_loads_eagerly_with_pid() {
        let text_calls = Arc::new(Mutex::new(0usize));
        let heavy_calls = Arc::new(Mutex::new(0usize));
        let (lt, lh) = counting_loaders(&text_calls, &heavy_calls);
        let res = Residency::from_policy(OffloadPolicy::Resident, lt, lh).unwrap();
        assert!(
            !res.is_sequential(),
            "Resident policy must build a Resident residency"
        );
        assert_eq!(
            *text_calls.lock().unwrap(),
            1,
            "Resident loads the text encoder once, eagerly"
        );
        assert_eq!(
            *heavy_calls.lock().unwrap(),
            1,
            "Resident loads the heavy bundle once, eagerly"
        );
    }

    #[test]
    fn with_resident_parts_borrows_under_resident_and_is_none_under_sequential() {
        // The read-only accessor providers use to reach warm components outside `run` (Chroma's
        // `*_ref`/`denoise` real-weight helpers): `Some((text, heavy))` under `Resident`, `None`
        // under `Sequential` (which holds no components between generates).
        let log: Log = new_log();
        let resident = Residency::resident(
            FakeText { log: log.clone() },
            FakeHeavy {
                log: log.clone(),
                with_pid: true,
            },
        );
        assert!(
            resident.with_resident_parts(|_, _| ()).is_some(),
            "Resident must expose its warm components"
        );
        let sequential = seq_residency(&log);
        assert!(
            sequential.with_resident_parts(|_, _| ()).is_none(),
            "Sequential holds no warm components to borrow"
        );
    }

    #[test]
    fn from_policy_resident_heavy_gets_use_pid_true() {
        // Resident loads any PiD overlay once (use_pid=true), reused across generates (F-177 skip is
        // Sequential-only). Assert the heavy build carried the PiD flag.
        let log = new_log();
        let lh_log = log.clone();
        let res = Residency::from_policy(
            OffloadPolicy::Resident,
            {
                let lt = log.clone();
                move || Ok(FakeText { log: lt.clone() })
            },
            move |use_pid| {
                Ok(FakeHeavy {
                    log: lh_log.clone(),
                    with_pid: use_pid,
                })
            },
        )
        .unwrap();
        // Drive a run and inspect the heavy's recorded drop tag (pid vs nopid).
        res.run(
            &CancelFlag::new(),
            false,
            &mut ignore_progress,
            |_t| Ok(()),
            |_e| Ok(()),
            |_h, _e, _op| Ok(()),
        )
        .unwrap();
        // Resident holds its components through `run`; the heavy's drop tag (pid vs nopid) only lands
        // once the residency itself is dropped. Drop it, then inspect the recorded tag.
        drop(res);
        assert!(
            events(&log).iter().any(|e| e == "drop_heavy_pid"),
            "Resident heavy must be built with use_pid=true: {:?}",
            events(&log)
        );
    }
}
