//! Backend-neutral component residency and phase-order driver.
//!
//! Image backends differ in tensor materialization and allocator cleanup, but not in the ownership
//! rule: retain reloadable phase loaders, optionally cache one warm text/heavy pair, and when a
//! request selects staged residency run text before heavy without overlapping their lifetimes.  This
//! module owns that shared state machine. Backend crates provide only [`ResidencyRuntime`] hooks and
//! their tensor phase bodies.

use crate::{CancelFlag, Error, LoadPhase, OffloadPolicy, Progress};
use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};

/// Backend behavior at a component-release boundary.
///
/// MLX flushes its allocator cache after a component drops. Candle deliberately does nothing here:
/// its device synchronization is a fallible, request-local operation supplied as the
/// `materialize_before_text_drop` closure to [`Residency::run_request_scoped`].
pub trait ResidencyRuntime {
    type Error: From<Error>;

    fn after_component_drop();
}

type RuntimeResult<R, T> = std::result::Result<T, <R as ResidencyRuntime>::Error>;
type TextLoader<Text, R> =
    Box<dyn Fn(bool, Option<&CancelFlag>) -> RuntimeResult<R, Text> + Send + Sync>;
type HeavyLoader<Heavy, R> =
    Box<dyn Fn(bool, bool, Option<&CancelFlag>) -> RuntimeResult<R, Heavy> + Send + Sync>;
type WarmLoader<Text, Heavy, R> =
    Box<dyn Fn(bool) -> RuntimeResult<R, (Text, Heavy)> + Send + Sync>;

struct ResidentPair<Text, Heavy> {
    text: Text,
    heavy: Heavy,
    streamable: bool,
}

struct Loaders<Text, Heavy, R: ResidencyRuntime> {
    load_text: TextLoader<Text, R>,
    load_heavy: HeavyLoader<Heavy, R>,
    load_warm: Option<WarmLoader<Text, Heavy, R>>,
}

/// A heavy render bundle whose denoise-only component can be shed before decode.
pub trait StagedHeavy {
    type Light;
    type DecodeView<'a>
    where
        Self: 'a,
        Self::Light: 'a;

    fn shed_dit(self) -> Self::Light;
    fn decode_view(&self) -> Self::DecodeView<'_>;
    fn light_view(light: &Self::Light) -> Self::DecodeView<'_>;
}

/// Shared ownership and request-scoped scheduling for a phase-A component and heavy render bundle.
pub struct Residency<Text, Heavy, R: ResidencyRuntime> {
    loaders: Loaders<Text, Heavy, R>,
    warm: Mutex<Option<ResidentPair<Text, Heavy>>>,
    rebuildable: bool,
    default_stage_residency: bool,
    on_staged: Option<Box<dyn Fn() + Send + Sync>>,
    runtime: PhantomData<R>,
}

impl<Text, Heavy, R: ResidencyRuntime> Residency<Text, Heavy, R> {
    /// Construct an in-memory resident-only value. It cannot rebuild after a staged request.
    pub fn resident(text: Text, heavy: Heavy) -> Self {
        Self {
            loaders: Loaders {
                load_text: Box::new(|_, _| {
                    Err(Error::Msg("resident-only source cannot reload text".into()).into())
                }),
                load_heavy: Box::new(|_, _, _| {
                    Err(
                        Error::Msg("resident-only source cannot reload heavy components".into())
                            .into(),
                    )
                }),
                load_warm: None,
            },
            warm: Mutex::new(Some(ResidentPair {
                text,
                heavy,
                streamable: false,
            })),
            rebuildable: false,
            default_stage_residency: false,
            on_staged: None,
            runtime: PhantomData,
        }
    }

    /// Compatibility constructor for fixed sequential-policy providers.
    pub fn sequential(
        load_text: impl Fn() -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: Loaders {
                load_text: Box::new(move |_, _| load_text()),
                load_heavy: Box::new(move |use_pid, _, _| load_heavy(use_pid)),
                load_warm: None,
            },
            warm: Mutex::new(None),
            rebuildable: true,
            default_stage_residency: true,
            on_staged: None,
            runtime: PhantomData,
        }
    }

    /// Construct request-scoped residency from independent text and heavy loaders.
    pub fn request_scoped(
        load_text: impl Fn(bool) -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool, bool) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: Loaders {
                load_text: Box::new(move |streamable, _| load_text(streamable)),
                load_heavy: Box::new(move |use_pid, streamable, _| load_heavy(use_pid, streamable)),
                load_warm: None,
            },
            warm: Mutex::new(None),
            rebuildable: true,
            default_stage_residency: false,
            on_staged: None,
            runtime: PhantomData,
        }
    }

    /// Construct request-scoped residency for a provider whose warm aggregate has a distinct loader.
    pub fn request_scoped_with_resident(
        load_warm: impl Fn(bool) -> RuntimeResult<R, (Text, Heavy)> + Send + Sync + 'static,
        load_text: impl Fn(bool) -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool, bool) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: Loaders {
                load_text: Box::new(move |streamable, _| load_text(streamable)),
                load_heavy: Box::new(move |use_pid, streamable, _| load_heavy(use_pid, streamable)),
                load_warm: Some(Box::new(load_warm)),
            },
            warm: Mutex::new(None),
            rebuildable: true,
            default_stage_residency: false,
            on_staged: None,
            runtime: PhantomData,
        }
    }

    /// Construct request-scoped residency whose staged loaders can observe the active request's
    /// cancellation flag. The distinct warm aggregate remains request-independent; only loads that
    /// occur inside [`Self::run_request_scoped`] receive the flag.
    pub fn request_scoped_with_resident_cancelable(
        load_warm: impl Fn(bool) -> RuntimeResult<R, (Text, Heavy)> + Send + Sync + 'static,
        load_text: impl Fn(bool, &CancelFlag) -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool, bool, &CancelFlag) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loaders: Loaders {
                load_text: Box::new(move |streamable, cancel| {
                    let Some(cancel) = cancel else {
                        return Err(Into::<R::Error>::into(Error::Msg(
                            "cancel-aware text loader used outside a request-scoped staged load"
                                .into(),
                        )));
                    };
                    load_text(streamable, cancel)
                }),
                load_heavy: Box::new(move |use_pid, streamable, cancel| {
                    let Some(cancel) = cancel else {
                        return Err(Into::<R::Error>::into(Error::Msg(
                            "cancel-aware heavy loader used outside a request-scoped staged load"
                                .into(),
                        )));
                    };
                    load_heavy(use_pid, streamable, cancel)
                }),
                load_warm: Some(Box::new(load_warm)),
            },
            warm: Mutex::new(None),
            rebuildable: true,
            default_stage_residency: false,
            on_staged: None,
            runtime: PhantomData,
        }
    }

    /// Compatibility dispatch for providers that still intentionally use a load-time policy.
    pub fn from_policy(
        policy: OffloadPolicy,
        load_text: impl Fn() -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> RuntimeResult<R, Self> {
        match policy {
            OffloadPolicy::Resident => {
                let residency = Self::request_scoped(
                    move |_| load_text(),
                    move |use_pid, _| load_heavy(use_pid),
                );
                {
                    let mut warm = residency.warm()?;
                    residency.ensure_warm_locked(&mut warm, false, &mut |_| {})?;
                }
                Ok(residency)
            }
            OffloadPolicy::Sequential => Ok(Self::sequential(load_text, load_heavy)),
        }
    }

    /// Compatibility dispatch for a distinct, lazily loaded warm aggregate.
    pub fn from_policy_with_resident(
        policy: OffloadPolicy,
        load_resident: impl Fn() -> RuntimeResult<R, (Text, Heavy)> + Send + Sync + 'static,
        load_text: impl Fn() -> RuntimeResult<R, Text> + Send + Sync + 'static,
        load_heavy: impl Fn(bool) -> RuntimeResult<R, Heavy> + Send + Sync + 'static,
    ) -> RuntimeResult<R, Self> {
        match policy {
            OffloadPolicy::Resident => {
                let mut residency = Self::request_scoped_with_resident(
                    move |_| load_resident(),
                    move |_| load_text(),
                    move |use_pid, _| load_heavy(use_pid),
                );
                residency.default_stage_residency = false;
                Ok(residency)
            }
            OffloadPolicy::Sequential => Ok(Self::sequential(load_text, load_heavy)),
        }
    }

    pub fn with_staged_advisory(mut self, advisory: impl Fn() + Send + Sync + 'static) -> Self {
        self.on_staged = Some(Box::new(advisory));
        self
    }

    /// Legacy fixed-policy discriminator. Request-scoped callers pass their selection explicitly.
    pub fn is_sequential(&self) -> bool {
        self.default_stage_residency
    }

    pub fn with_resident_parts<T>(
        &self,
        f: impl FnOnce(&Text, &Heavy) -> T,
    ) -> RuntimeResult<R, Option<T>> {
        let warm = self.warm()?;
        Ok(warm.as_ref().map(|pair| f(&pair.text, &pair.heavy)))
    }

    /// Evict a warm pair before a backend-specific multi-stage path starts loading its first phase.
    /// Ordinary two-phase callers should use [`run_request_scoped`](Self::run_request_scoped), which
    /// keeps the lifecycle lock for the whole request. This escape hatch is for already-established
    /// serialized video-like phase bodies whose order is intentionally not collapsed into this seam.
    pub fn evict_warm(&self) -> RuntimeResult<R, bool> {
        self.ensure_rebuildable("evict warm components")?;
        let mut warm = self.warm()?;
        Ok(Self::evict_warm_locked(&mut warm))
    }

    fn warm(&self) -> RuntimeResult<R, MutexGuard<'_, Option<ResidentPair<Text, Heavy>>>> {
        self.warm
            .lock()
            .map_err(|_| Error::Msg("component-residency mutex poisoned".into()).into())
    }

    fn ensure_rebuildable(&self, action: &str) -> RuntimeResult<R, ()> {
        if self.rebuildable {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "resident-only component source cannot {action} because it has no reload loaders"
            ))
            .into())
        }
    }

    fn evict_warm_locked(warm: &mut Option<ResidentPair<Text, Heavy>>) -> bool {
        let Some(pair) = warm.take() else {
            return false;
        };
        drop(pair);
        R::after_component_drop();
        true
    }

    fn ensure_warm_locked(
        &self,
        warm: &mut Option<ResidentPair<Text, Heavy>>,
        streamable: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> RuntimeResult<R, ()> {
        if !self.rebuildable && (streamable || warm.is_none()) {
            return Err(Error::Unsupported(
                "resident-only component source cannot change materialization shape or reload"
                    .into(),
            )
            .into());
        }
        if warm
            .as_ref()
            .is_some_and(|pair| pair.streamable != streamable)
        {
            Self::evict_warm_locked(warm);
        }
        if warm.is_some() {
            return Ok(());
        }

        let (text, heavy) = if let Some(load_warm) = &self.loaders.load_warm {
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            match load_warm(streamable) {
                Ok(pair) => pair,
                Err(error) => {
                    R::after_component_drop();
                    return Err(error);
                }
            }
        } else {
            on_progress(Progress::Loading(LoadPhase::TextEncoder));
            let text = match (self.loaders.load_text)(streamable, None) {
                Ok(text) => text,
                Err(error) => {
                    R::after_component_drop();
                    return Err(error);
                }
            };
            on_progress(Progress::Loading(LoadPhase::Renderer));
            // Warm residents deliberately load the reusable PiD superset once. Per-request
            // `use_pid` selects the decode path; it must not narrow the cross-request warm pair.
            let heavy = match (self.loaders.load_heavy)(true, streamable, None) {
                Ok(heavy) => heavy,
                Err(error) => {
                    drop(text);
                    R::after_component_drop();
                    return Err(error);
                }
            };
            (text, heavy)
        };
        *warm = Some(ResidentPair {
            text,
            heavy,
            streamable,
        });
        Ok(())
    }

    /// Compatibility entry point for fixed-policy providers.
    #[allow(clippy::too_many_arguments)]
    pub fn run<Encoded, Out>(
        &self,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> RuntimeResult<R, Encoded>,
        materialize_before_text_drop: impl FnOnce(Option<&Encoded>) -> RuntimeResult<R, ()>,
        render: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> RuntimeResult<R, Out>,
    ) -> RuntimeResult<R, Out> {
        self.run_request_scoped(
            self.default_stage_residency,
            false,
            cancel,
            use_pid,
            on_progress,
            encode,
            materialize_before_text_drop,
            render,
        )
    }

    /// Execute one request. `stage_residency` is the sole production lifecycle authority.
    #[allow(clippy::too_many_arguments)]
    pub fn run_request_scoped<Encoded, Out>(
        &self,
        stage_residency: bool,
        streamable: bool,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> RuntimeResult<R, Encoded>,
        materialize_before_text_drop: impl FnOnce(Option<&Encoded>) -> RuntimeResult<R, ()>,
        render: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> RuntimeResult<R, Out>,
    ) -> RuntimeResult<R, Out> {
        check_cancel::<R>(cancel)?;
        if stage_residency || streamable {
            self.ensure_rebuildable("stage or rematerialize components")?;
        }
        let mut warm = self.warm()?;
        if !stage_residency {
            self.ensure_warm_locked(&mut warm, streamable, on_progress)?;
            let pair = warm
                .as_ref()
                .ok_or_else(|| Error::Msg("warm residency was not populated".into()))
                .map_err(Into::<R::Error>::into)?;
            let encoded = match encode(&pair.text) {
                Ok(encoded) => encoded,
                Err(error) => {
                    let _ = materialize_before_text_drop(None);
                    return Err(error);
                }
            };
            check_cancel::<R>(cancel)?;
            let result = render(&pair.heavy, encoded, on_progress);
            drop(warm);
            return result;
        }

        if let Some(advisory) = &self.on_staged {
            advisory();
        }
        Self::evict_warm_locked(&mut warm);
        let result = run_two_phase::<R, _, _, _, _>(
            cancel,
            on_progress,
            || (self.loaders.load_text)(streamable, Some(cancel)),
            encode,
            materialize_before_text_drop,
            || (self.loaders.load_heavy)(use_pid, streamable, Some(cancel)),
            render,
        );
        drop(warm);
        result
    }

    /// Run an established backend-specific staged body while keeping this owner's lifecycle lock.
    ///
    /// This is intentionally not a second phase-order driver: the backend closure retains its
    /// genuine three-or-more phase schedule. The shared owner only guarantees that a concurrent warm
    /// request cannot repopulate the evicted pair while that schedule is in flight.
    pub fn run_exclusive_staged<Out>(
        &self,
        cancel: &CancelFlag,
        body: impl FnOnce() -> RuntimeResult<R, Out>,
    ) -> RuntimeResult<R, Out> {
        check_cancel::<R>(cancel)?;
        self.ensure_rebuildable("enter an exclusive staged request")?;
        let mut warm = self.warm()?;
        Self::evict_warm_locked(&mut warm);
        let result = body();
        drop(warm);
        result
    }
}

impl<Text, Heavy: StagedHeavy, R: ResidencyRuntime> Residency<Text, Heavy, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn run_staged<Encoded, Mid, Out>(
        &self,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> RuntimeResult<R, Encoded>,
        materialize_encoded: impl FnOnce(Option<&Encoded>) -> RuntimeResult<R, ()>,
        denoise: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> RuntimeResult<R, Mid>,
        materialize_mid: impl FnOnce(&Mid) -> RuntimeResult<R, ()>,
        decode: impl FnOnce(
            Heavy::DecodeView<'_>,
            Mid,
            &mut dyn FnMut(Progress),
        ) -> RuntimeResult<R, Out>,
    ) -> RuntimeResult<R, Out> {
        self.run_staged_request_scoped(
            self.default_stage_residency,
            false,
            cancel,
            use_pid,
            on_progress,
            encode,
            materialize_encoded,
            denoise,
            materialize_mid,
            decode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_staged_request_scoped<Encoded, Mid, Out>(
        &self,
        stage_residency: bool,
        streamable: bool,
        cancel: &CancelFlag,
        use_pid: bool,
        on_progress: &mut dyn FnMut(Progress),
        encode: impl FnOnce(&Text) -> RuntimeResult<R, Encoded>,
        materialize_encoded: impl FnOnce(Option<&Encoded>) -> RuntimeResult<R, ()>,
        denoise: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> RuntimeResult<R, Mid>,
        materialize_mid: impl FnOnce(&Mid) -> RuntimeResult<R, ()>,
        decode: impl FnOnce(
            Heavy::DecodeView<'_>,
            Mid,
            &mut dyn FnMut(Progress),
        ) -> RuntimeResult<R, Out>,
    ) -> RuntimeResult<R, Out> {
        check_cancel::<R>(cancel)?;
        if stage_residency || streamable {
            self.ensure_rebuildable("stage or rematerialize components")?;
        }
        let mut warm = self.warm()?;
        if !stage_residency {
            self.ensure_warm_locked(&mut warm, streamable, on_progress)?;
            let pair = warm
                .as_ref()
                .ok_or_else(|| Error::Msg("warm residency was not populated".into()))
                .map_err(Into::<R::Error>::into)?;
            let encoded = match encode(&pair.text) {
                Ok(encoded) => encoded,
                Err(error) => {
                    let _ = materialize_encoded(None);
                    return Err(error);
                }
            };
            check_cancel::<R>(cancel)?;
            let mid = denoise(&pair.heavy, encoded, on_progress)?;
            let result = decode(pair.heavy.decode_view(), mid, on_progress);
            drop(warm);
            return result;
        }

        if let Some(advisory) = &self.on_staged {
            advisory();
        }
        Self::evict_warm_locked(&mut warm);

        on_progress(Progress::Loading(LoadPhase::TextEncoder));
        let text = match (self.loaders.load_text)(streamable, Some(cancel)) {
            Ok(text) => text,
            Err(error) => {
                R::after_component_drop();
                return Err(error);
            }
        };
        let encoded = match encode(&text) {
            Ok(encoded) => materialize_encoded(Some(&encoded)).map(|()| encoded),
            Err(error) => {
                let _ = materialize_encoded(None);
                Err(error)
            }
        };
        drop(text);
        R::after_component_drop();
        let encoded = encoded?;
        check_cancel::<R>(cancel)?;

        on_progress(Progress::Loading(LoadPhase::Renderer));
        let heavy = match (self.loaders.load_heavy)(use_pid, streamable, Some(cancel)) {
            Ok(heavy) => heavy,
            Err(error) => {
                R::after_component_drop();
                return Err(error);
            }
        };
        let light_and_mid = check_cancel::<R>(cancel).and_then(|()| {
            denoise(&heavy, encoded, on_progress).and_then(|mid| {
                materialize_mid(&mid)?;
                Ok(mid)
            })
        });
        let light_and_mid = match light_and_mid {
            Ok(mid) => Ok((heavy.shed_dit(), mid)),
            Err(error) => {
                drop(heavy);
                Err(error)
            }
        };
        R::after_component_drop();
        let (light, mid) = light_and_mid?;
        let result = decode(Heavy::light_view(&light), mid, on_progress);
        drop(light);
        R::after_component_drop();
        drop(warm);
        result
    }
}

/// The one shared two-phase order driver used by both backend adapters.
#[allow(clippy::too_many_arguments)]
pub fn run_two_phase<R, Text, Heavy, Encoded, Out>(
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> RuntimeResult<R, Text>,
    encode: impl FnOnce(&Text) -> RuntimeResult<R, Encoded>,
    materialize_before_text_drop: impl FnOnce(Option<&Encoded>) -> RuntimeResult<R, ()>,
    load_heavy: impl FnOnce() -> RuntimeResult<R, Heavy>,
    render: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> RuntimeResult<R, Out>,
) -> RuntimeResult<R, Out>
where
    R: ResidencyRuntime,
{
    check_cancel::<R>(cancel)?;
    on_progress(Progress::Loading(LoadPhase::TextEncoder));
    let text = match load_text() {
        Ok(text) => text,
        Err(error) => {
            R::after_component_drop();
            return Err(error);
        }
    };
    let encoded = match encode(&text) {
        Ok(encoded) => materialize_before_text_drop(Some(&encoded)).map(|()| encoded),
        Err(error) => {
            let _ = materialize_before_text_drop(None);
            Err(error)
        }
    };
    drop(text);
    R::after_component_drop();
    let encoded = encoded?;

    check_cancel::<R>(cancel)?;
    on_progress(Progress::Loading(LoadPhase::Renderer));
    let heavy = match load_heavy() {
        Ok(heavy) => heavy,
        Err(error) => {
            R::after_component_drop();
            return Err(error);
        }
    };
    let result = check_cancel::<R>(cancel).and_then(|()| render(&heavy, encoded, on_progress));
    drop(heavy);
    R::after_component_drop();
    result
}

fn check_cancel<R: ResidencyRuntime>(cancel: &CancelFlag) -> RuntimeResult<R, ()> {
    if cancel.is_cancelled() {
        Err(Error::Canceled.into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static CLEANUPS: AtomicUsize = AtomicUsize::new(0);

    struct Runtime;
    impl ResidencyRuntime for Runtime {
        type Error = Error;
        fn after_component_drop() {
            CLEANUPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Witness {
        name: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
    }
    impl Drop for Witness {
        fn drop(&mut self) {
            self.log.lock().unwrap().push(self.name);
        }
    }

    #[test]
    fn warm_staged_warm_sequence_evicts_before_load_and_repopulates_later() {
        CLEANUPS.store(0, Ordering::SeqCst);
        let log = Arc::new(Mutex::new(Vec::new()));
        let text_log = Arc::clone(&log);
        let heavy_log = Arc::clone(&log);
        let residency = Residency::<Witness, Witness, Runtime>::request_scoped(
            move |_| {
                text_log.lock().unwrap().push("load-text");
                Ok(Witness {
                    name: "drop-text",
                    log: Arc::clone(&text_log),
                })
            },
            move |_, _| {
                heavy_log.lock().unwrap().push("load-heavy");
                Ok(Witness {
                    name: "drop-heavy",
                    log: Arc::clone(&heavy_log),
                })
            },
        );
        let cancel = CancelFlag::new();

        let run = |stage, log: &Arc<Mutex<Vec<&'static str>>>| {
            residency.run_request_scoped(
                stage,
                false,
                &cancel,
                false,
                &mut |_| {},
                |_| {
                    log.lock().unwrap().push("encode");
                    Ok(())
                },
                |_| {
                    log.lock().unwrap().push("materialize");
                    Ok(())
                },
                |_, _, _| {
                    log.lock().unwrap().push("render");
                    Ok(())
                },
            )
        };

        run(false, &log).unwrap();
        run(true, &log).unwrap();
        run(false, &log).unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                "load-text",
                "load-heavy",
                "encode",
                "render",
                "drop-text",
                "drop-heavy",
                "load-text",
                "encode",
                "materialize",
                "drop-text",
                "load-heavy",
                "render",
                "drop-heavy",
                "load-text",
                "load-heavy",
                "encode",
                "render",
            ]
        );
    }

    #[test]
    fn resident_only_rejects_eviction_before_losing_its_warm_pair() {
        let residency = Residency::<u8, u8, Runtime>::resident(2, 3);
        let callbacks = AtomicUsize::new(0);
        let run = |stage_residency, streamable| {
            residency.run_request_scoped(
                stage_residency,
                streamable,
                &CancelFlag::new(),
                false,
                &mut |_| {
                    callbacks.fetch_add(1, Ordering::SeqCst);
                },
                |text| {
                    callbacks.fetch_add(1, Ordering::SeqCst);
                    Ok(*text)
                },
                |_| {
                    callbacks.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                |heavy, encoded, _| {
                    callbacks.fetch_add(1, Ordering::SeqCst);
                    Ok(*heavy + encoded)
                },
            )
        };

        for (stage_residency, streamable) in [(true, false), (false, true)] {
            let error: Error = run(stage_residency, streamable).unwrap_err();
            assert!(
                matches!(error, Error::Unsupported(message) if message.contains("resident-only"))
            );
            assert_eq!(callbacks.load(Ordering::SeqCst), 0);
        }
        let error: Error = residency.evict_warm().unwrap_err();
        assert!(matches!(error, Error::Unsupported(message) if message.contains("resident-only")));
        let body_calls = AtomicUsize::new(0);
        let error: Error = residency
            .run_exclusive_staged(&CancelFlag::new(), || {
                body_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(message) if message.contains("resident-only")));
        assert_eq!(body_calls.load(Ordering::SeqCst), 0);

        assert_eq!(run(false, false).unwrap(), 5);
        assert_eq!(callbacks.load(Ordering::SeqCst), 2);
        assert_eq!(
            residency
                .with_resident_parts(|text, heavy| *text + *heavy)
                .unwrap(),
            Some(5)
        );
    }

    #[test]
    fn every_residency_accessor_fails_closed_after_mutex_poisoning() {
        let residency = Residency::<u8, u8, Runtime>::request_scoped(|_| Ok(2), |_, _| Ok(3));
        let ordinary_run = || {
            residency.run_request_scoped(
                false,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
        };
        assert_eq!(ordinary_run().unwrap(), 5);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = residency.warm.lock().unwrap();
            panic!("poison residency for the fail-closed policy test");
        }));

        let assert_poison = |error: Error| {
            assert!(
                matches!(error, Error::Msg(message) if message == "component-residency mutex poisoned")
            );
        };
        assert_poison(residency.with_resident_parts(|_, _| ()).unwrap_err());
        assert_poison(ordinary_run().unwrap_err());
        assert_poison(residency.evict_warm().unwrap_err());
    }

    #[test]
    fn staged_cancel_after_encode_releases_text_and_never_loads_heavy() {
        CLEANUPS.store(0, Ordering::SeqCst);
        let heavy_loads = Arc::new(AtomicUsize::new(0));
        let observed_heavy_loads = Arc::clone(&heavy_loads);
        let residency = Residency::<u8, u8, Runtime>::request_scoped(
            |_| Ok(2),
            move |_, _| {
                observed_heavy_loads.fetch_add(1, Ordering::SeqCst);
                Ok(3)
            },
        );
        let cancel = CancelFlag::new();
        let cancel_at_boundary = cancel.clone();
        let result: Result<(), Error> = residency.run_request_scoped(
            true,
            false,
            &cancel,
            false,
            &mut |_| {},
            |_| Ok(4),
            |_| {
                cancel_at_boundary.cancel();
                Ok(())
            },
            |_, _, _| Ok(()),
        );

        assert!(matches!(result, Err(Error::Canceled)));
        assert_eq!(heavy_loads.load(Ordering::SeqCst), 0);
        assert_eq!(CLEANUPS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn lazy_resident_compatibility_loader_stays_warm() {
        let warm_loads = Arc::new(AtomicUsize::new(0));
        let observed_warm_loads = Arc::clone(&warm_loads);
        let residency = Residency::<u8, u8, Runtime>::from_policy_with_resident(
            OffloadPolicy::Resident,
            move || {
                observed_warm_loads.fetch_add(1, Ordering::SeqCst);
                Ok((2, 3))
            },
            || Ok(20),
            |_| Ok(30),
        )
        .unwrap();
        assert!(!residency.is_sequential());
        assert_eq!(warm_loads.load(Ordering::SeqCst), 0);

        let run = || {
            residency.run(
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
        };
        assert_eq!(run().unwrap(), 5);
        assert_eq!(run().unwrap(), 5);
        assert_eq!(warm_loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn eager_resident_policy_retains_loaders_for_a_later_staged_request() {
        let text_loads = Arc::new(AtomicUsize::new(0));
        let heavy_loads = Arc::new(AtomicUsize::new(0));
        let observed_text = Arc::clone(&text_loads);
        let observed_heavy = Arc::clone(&heavy_loads);
        let residency = Residency::<u8, u8, Runtime>::from_policy(
            OffloadPolicy::Resident,
            move || {
                observed_text.fetch_add(1, Ordering::SeqCst);
                Ok(2)
            },
            move |_| {
                observed_heavy.fetch_add(1, Ordering::SeqCst);
                Ok(3)
            },
        )
        .unwrap();
        assert_eq!(text_loads.load(Ordering::SeqCst), 1);
        assert_eq!(heavy_loads.load(Ordering::SeqCst), 1);

        let output = residency
            .run_request_scoped(
                true,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(()),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
            .unwrap();
        assert_eq!(output, 5);
        assert_eq!(text_loads.load(Ordering::SeqCst), 2);
        assert_eq!(heavy_loads.load(Ordering::SeqCst), 2);
    }

    static WARM_FAILURE_CLEANUPS: AtomicUsize = AtomicUsize::new(0);
    struct WarmFailureRuntime;
    impl ResidencyRuntime for WarmFailureRuntime {
        type Error = Error;
        fn after_component_drop() {
            WARM_FAILURE_CLEANUPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn warm_heavy_load_failure_releases_the_partial_text_phase() {
        WARM_FAILURE_CLEANUPS.store(0, Ordering::SeqCst);
        let residency = Residency::<u8, u8, WarmFailureRuntime>::request_scoped(
            |_| Ok(2),
            |_, _| Err(Error::Msg("heavy failed".into())),
        );
        let result: Result<(), Error> = residency.run_request_scoped(
            false,
            false,
            &CancelFlag::new(),
            false,
            &mut |_| {},
            |_| Ok(()),
            |_| Ok(()),
            |_, _, _| Ok(()),
        );
        assert!(matches!(result, Err(Error::Msg(ref message)) if message == "heavy failed"));
        assert_eq!(WARM_FAILURE_CLEANUPS.load(Ordering::SeqCst), 1);
    }

    trait TraceRuntime: ResidencyRuntime<Error = Error> {
        fn trace() -> &'static Mutex<Vec<&'static str>>;
    }

    struct CandleTraceRuntime;
    static CANDLE_TRACE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    impl ResidencyRuntime for CandleTraceRuntime {
        type Error = Error;
        fn after_component_drop() {
            CANDLE_TRACE.lock().unwrap().push("release");
        }
    }
    impl TraceRuntime for CandleTraceRuntime {
        fn trace() -> &'static Mutex<Vec<&'static str>> {
            &CANDLE_TRACE
        }
    }

    struct MlxTraceRuntime;
    static MLX_TRACE: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    impl ResidencyRuntime for MlxTraceRuntime {
        type Error = Error;
        fn after_component_drop() {
            MLX_TRACE.lock().unwrap().push("release");
        }
    }
    impl TraceRuntime for MlxTraceRuntime {
        fn trace() -> &'static Mutex<Vec<&'static str>> {
            &MLX_TRACE
        }
    }

    struct TraceWitness<R: TraceRuntime> {
        drop_event: &'static str,
        runtime: PhantomData<R>,
    }
    impl<R: TraceRuntime> Drop for TraceWitness<R> {
        fn drop(&mut self) {
            R::trace().lock().unwrap().push(self.drop_event);
        }
    }

    fn staged_trace<R: TraceRuntime>() -> Vec<&'static str> {
        R::trace().lock().unwrap().clear();
        run_two_phase::<R, _, _, _, _>(
            &CancelFlag::new(),
            &mut |progress| match progress {
                Progress::Loading(LoadPhase::TextEncoder) => {
                    R::trace().lock().unwrap().push("load-text")
                }
                Progress::Loading(LoadPhase::Renderer) => {
                    R::trace().lock().unwrap().push("load-heavy")
                }
                _ => {}
            },
            || {
                Ok(TraceWitness::<R> {
                    drop_event: "drop-text",
                    runtime: PhantomData,
                })
            },
            |_| {
                R::trace().lock().unwrap().push("encode");
                Ok(())
            },
            |_| {
                R::trace().lock().unwrap().push("materialize");
                Ok(())
            },
            || {
                Ok(TraceWitness::<R> {
                    drop_event: "drop-heavy",
                    runtime: PhantomData,
                })
            },
            |_, _, _| {
                R::trace().lock().unwrap().push("render");
                Ok(())
            },
        )
        .unwrap();
        R::trace().lock().unwrap().clone()
    }

    #[test]
    fn both_backend_adapters_share_phase_order_and_release_points() {
        let candle = staged_trace::<CandleTraceRuntime>();
        let mlx = staged_trace::<MlxTraceRuntime>();
        assert_eq!(candle, mlx);
        assert_eq!(
            candle,
            vec![
                "load-text",
                "encode",
                "materialize",
                "drop-text",
                "release",
                "load-heavy",
                "render",
                "drop-heavy",
                "release",
            ]
        );
    }
}
