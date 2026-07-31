//! Candle adapter for gen-core's backend-neutral request-scoped residency state machine.
//!
//! The shared core owns warm-cache transitions and the image text→heavy phase order. Candle supplies
//! only the mandatory post-encode device synchronization; its allocator intentionally has no MLX-like
//! cache flush. The separate three-component video driver remains here because its phase bodies and
//! cancellation/progress policy are owned by those video pipelines.

use candle_core::Device;
use gen_core::{CancelFlag, Progress};

use crate::{CandleError, Result};

/// Candle release behavior for the shared residency driver.
pub struct CandleResidencyRuntime;

impl gen_core::ResidencyRuntime for CandleResidencyRuntime {
    type Error = CandleError;

    fn after_component_drop() {}
}

/// The one shared residency implementation, specialized only by Candle cleanup behavior.
pub type Residency<Text, Heavy> = gen_core::Residency<Text, Heavy, CandleResidencyRuntime>;

pub use gen_core::StagedHeavy;

/// Return a typed cancellation error when the request flag is tripped.
pub fn check_cancel(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(CandleError::Canceled)
    } else {
        Ok(())
    }
}

/// Compatibility entry point for video/bespoke callers that do not retain a [`Residency`].
/// The phase order itself is the same gen-core driver used by both backend aliases.
pub fn run_sequential<Text, Heavy, Encoded, Out>(
    cancel: &CancelFlag,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> Result<Text>,
    encode: impl FnOnce(&Text) -> Result<Encoded>,
    load_heavy: impl FnOnce() -> Result<Heavy>,
    render: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    run_sequential_with_sync(
        cancel,
        on_progress,
        load_text,
        encode,
        || Ok(device.synchronize()?),
        load_heavy,
        |heavy, encoded, on_progress| {
            let result = render(heavy, encoded, on_progress);
            synchronize_result(device, result)
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn run_sequential_with_sync<Text, Heavy, Encoded, Out>(
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    load_text: impl FnOnce() -> Result<Text>,
    encode: impl FnOnce(&Text) -> Result<Encoded>,
    sync_text_boundary: impl FnOnce() -> Result<()>,
    load_heavy: impl FnOnce() -> Result<Heavy>,
    render: impl FnOnce(&Heavy, Encoded, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    gen_core::residency::run_two_phase::<CandleResidencyRuntime, _, _, _, _>(
        cancel,
        on_progress,
        load_text,
        encode,
        |_| sync_text_boundary(),
        load_heavy,
        render,
    )
}

/// Synchronize queued device work before the component borrowed by `result`'s producer is dropped.
/// Preserve the phase body's primary error when both the work and synchronization fail.
pub fn synchronize_result<T>(device: &Device, result: Result<T>) -> Result<T> {
    let synchronized = device.synchronize().map_err(CandleError::from);
    finish_synchronized_phase(result, synchronized)
}

/// Drive a video-specific three-component lifecycle while keeping at most one component resident.
#[allow(clippy::too_many_arguments)]
pub fn run_three_stage_sequential<A, B, C, State, Out>(
    state: &mut State,
    load_a: impl FnOnce(&mut State) -> Result<A>,
    use_a: impl FnOnce(&A, &mut State) -> Result<()>,
    load_b: impl FnOnce(&mut State) -> Result<B>,
    use_b: impl FnOnce(&B, &mut State) -> Result<()>,
    load_c: impl FnOnce(&mut State) -> Result<C>,
    use_c: impl FnOnce(&C, &mut State) -> Result<Out>,
    mut sync: impl FnMut() -> Result<()>,
) -> Result<Out> {
    {
        let phase = match load_a(state) {
            Ok(phase) => phase,
            Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
        };
        let used = use_a(&phase, state);
        let synced = sync();
        finish_synchronized_phase(used, synced)?;
    }
    {
        let phase = match load_b(state) {
            Ok(phase) => phase,
            Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
        };
        let used = use_b(&phase, state);
        let synced = sync();
        finish_synchronized_phase(used, synced)?;
    }
    let phase = match load_c(state) {
        Ok(phase) => phase,
        Err(primary) => return finish_synchronized_phase(Err(primary), sync()),
    };
    let used = use_c(&phase, state);
    let synced = sync();
    finish_synchronized_phase(used, synced)
}

fn finish_synchronized_phase<T>(used: Result<T>, synced: Result<()>) -> Result<T> {
    match used {
        Err(primary) => Err(primary),
        Ok(value) => {
            synced?;
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::LoadPhase;
    use std::cell::RefCell;

    struct Witness<'a> {
        name: &'static str,
        log: &'a RefCell<Vec<&'static str>>,
    }
    impl Drop for Witness<'_> {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.name);
        }
    }

    #[test]
    fn shared_two_phase_driver_syncs_and_drops_before_heavy_load() {
        let log = RefCell::new(Vec::new());
        let mut phases = Vec::new();
        let output = run_sequential(
            &CancelFlag::new(),
            &Device::Cpu,
            &mut |progress| {
                if let Progress::Loading(phase) = progress {
                    phases.push(phase);
                }
            },
            || {
                Ok(Witness {
                    name: "drop-text",
                    log: &log,
                })
            },
            |_| {
                log.borrow_mut().push("encode");
                Ok(2u8)
            },
            || {
                log.borrow_mut().push("load-heavy");
                Ok(3u8)
            },
            |_, encoded, _| {
                log.borrow_mut().push("render");
                Ok(encoded + 1)
            },
        )
        .unwrap();

        assert_eq!(output, 3);
        assert_eq!(phases, vec![LoadPhase::TextEncoder, LoadPhase::Renderer]);
        assert_eq!(
            *log.borrow(),
            vec!["encode", "drop-text", "load-heavy", "render"]
        );
    }

    #[test]
    fn encode_error_synchronizes_before_text_drop_and_skips_heavy_load() {
        let log = RefCell::new(Vec::new());
        let result: Result<()> = run_sequential_with_sync(
            &CancelFlag::new(),
            &mut |_| {},
            || {
                Ok(Witness {
                    name: "drop-text",
                    log: &log,
                })
            },
            |_| {
                log.borrow_mut().push("encode-error");
                Err::<u8, _>(CandleError::Msg("encode failed".into()))
            },
            || {
                log.borrow_mut().push("sync");
                Ok(())
            },
            || {
                log.borrow_mut().push("load-heavy");
                Ok(())
            },
            |_, _, _| Ok(()),
        );
        assert!(matches!(result, Err(CandleError::Msg(ref message)) if message == "encode failed"));
        assert_eq!(*log.borrow(), vec!["encode-error", "sync", "drop-text"]);
    }

    #[test]
    fn request_selection_stages_a_previously_warm_candle_residency() {
        let loads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let text_loads = std::sync::Arc::clone(&loads);
        let heavy_loads = std::sync::Arc::clone(&loads);
        let residency = Residency::request_scoped(
            move |_| {
                crate::lock_recover(&text_loads).push("text");
                Ok(2u8)
            },
            move |_, _| {
                crate::lock_recover(&heavy_loads).push("heavy");
                Ok(3u8)
            },
        );
        let run = |stage| {
            residency.run_request_scoped(
                stage,
                false,
                &CancelFlag::new(),
                false,
                &mut |_| {},
                |text| Ok(*text),
                |_| Ok(Device::Cpu.synchronize()?),
                |heavy, encoded, _| Ok(*heavy + encoded),
            )
        };

        assert_eq!(run(false).unwrap(), 5);
        assert_eq!(run(true).unwrap(), 5);
        assert_eq!(
            *crate::lock_recover(&loads),
            vec!["text", "heavy", "text", "heavy"]
        );
        assert!(residency.with_resident_parts(|_, _| ()).is_none());
    }

    #[test]
    fn three_stage_driver_keeps_video_phases_disjoint() {
        let log = RefCell::new(Vec::new());
        let mut state = ();
        let output = run_three_stage_sequential(
            &mut state,
            |_| {
                Ok(Witness {
                    name: "drop-a",
                    log: &log,
                })
            },
            |_, _| {
                log.borrow_mut().push("use-a");
                Ok(())
            },
            |_| {
                Ok(Witness {
                    name: "drop-b",
                    log: &log,
                })
            },
            |_, _| {
                log.borrow_mut().push("use-b");
                Ok(())
            },
            |_| {
                Ok(Witness {
                    name: "drop-c",
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

        assert_eq!(output, 7);
        assert_eq!(
            *log.borrow(),
            vec!["use-a", "sync", "drop-a", "use-b", "sync", "drop-b", "use-c", "sync", "drop-c"]
        );
    }
}
