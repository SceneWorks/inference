//! PREREQUISITE: validated mlx-rs retirement ABI descended from
//! 2f07b006f4a98660324382a8131bb5cdaf92e0b0, INCLUDING the subsequently added
//! safe Stream::get_device(), and its matching validated prebuilt. Never apply
//! or activate this increment against fe22998 or the getter-less 2f07 revision.
use super::*;
use std::cell::RefCell;
use std::rc::Rc;

pub fn retirement_label_id(label: &str) -> u64 {
    label
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

/// ALLOCATING metadata query. Use only before model/body builders, NEVER in an
/// OOM/error retirement cleanup. Drop the owned CDevice before any window/credit.
pub fn retirement_stream_is_cpu(stream: &mlx_rs::Stream) -> Result<bool> {
    let device = stream.get_device()?;
    let kind = device.get_type()?;
    drop(device);
    Ok(matches!(kind, mlx_rs::DeviceType::Cpu))
}
struct CachedStream {
    stream: mlx_rs::Stream,
    index: i32,
}
/// Prepare once BEFORE model load, retain across the loaded generator's life.
/// Handles/indices are cached; draining does not discover devices, allocate a
/// CDevice, create a fresh stream, query defaults, or install an error handler.
pub struct NativeRetirementStreams {
    streams: [Option<CachedStream>; 3],
    selected: i32,
    selected_cpu: bool,
    needs_gpu: bool,
}
impl NativeRetirementStreams {
    pub fn prepare() -> Result<Rc<Self>> {
        let selected = mlx_rs::Stream::task_local_or_default();
        let selected_index = selected.get_index()?;
        let selected_cpu = retirement_stream_is_cpu(&selected)?;
        let device = mlx_rs::Device::try_default()?;
        let default_cpu = matches!(device.get_type()?, mlx_rs::DeviceType::Cpu);
        drop(device);
        let needs_gpu = !selected_cpu || !default_cpu;
        let candidates = [
            needs_gpu.then(mlx_rs::Stream::gpu),
            Some(selected),
            Some(mlx_rs::Stream::cpu()),
        ];
        let mut streams: [Option<CachedStream>; 3] = [None, None, None];
        for (slot, stream) in candidates.into_iter().enumerate() {
            let Some(stream) = stream else {
                continue;
            };
            let index = stream.get_index()?;
            if index < 0 {
                return Err(Error::Unsupported(
                    "native retirement stream has invalid index".into(),
                ));
            }
            if streams[..slot]
                .iter()
                .flatten()
                .any(|known| known.index == index)
            {
                continue;
            }
            streams[slot] = Some(CachedStream { stream, index });
        }
        Ok(Rc::new(Self {
            streams,
            selected: selected_index,
            selected_cpu,
            needs_gpu,
        }))
    }
    fn observation(&self) -> [(&'static str, u64); 7] {
        let mask = self
            .streams
            .iter()
            .enumerate()
            .fold(0u64, |mask, (index, stream)| {
                mask | if stream.is_some() { 1 << index } else { 0 }
            });
        [
            ("selected_stream", self.selected as u64),
            ("stream_mask", mask),
            ("stream_count", mask.count_ones() as u64),
            (
                "stream_0",
                self.streams[0]
                    .as_ref()
                    .map_or(0, |stream| stream.index as u64),
            ),
            (
                "stream_1",
                self.streams[1]
                    .as_ref()
                    .map_or(0, |stream| stream.index as u64),
            ),
            (
                "stream_2",
                self.streams[2]
                    .as_ref()
                    .map_or(0, |stream| stream.index as u64),
            ),
            ("compute_cpu", u64::from(self.selected_cpu)),
        ]
    }
    /// Check BEFORE construction, not in cleanup. A changed default/task-local
    /// stream cannot silently escape the exact prepared relevant-stream set.
    pub fn validate_current(&self) -> Result<()> {
        if mlx_rs::Stream::task_local_or_default().get_index()? != self.selected {
            return Err(Error::Unsupported(
                "native retirement task-local stream differs from prepared lifecycle".into(),
            ));
        }
        let defaults = [
            self.needs_gpu.then(mlx_rs::Stream::gpu),
            Some(mlx_rs::Stream::cpu()),
        ];
        for stream in defaults.into_iter().flatten() {
            let index = stream.get_index()?;
            if !self
                .streams
                .iter()
                .flatten()
                .any(|known| known.index == index)
            {
                return Err(Error::Unsupported(
                    "native retirement default stream changed during lifecycle".into(),
                ));
            }
        }
        Ok(())
    }
    /// Try ALL cached relevant ranges on success/error/cancel. GPU producers
    /// precede CPU consumers; native capture also flushes mixed GPU producers
    /// before CPU joins. Its idle-buffer rule submits no idle GPU work.
    pub fn drain(&self) -> Result<()> {
        let mut failure = None;
        for known in self.streams.iter().flatten() {
            // Creation joins this captured range. Wait observes its durable
            // outcome; receipt Drop never acknowledges an unobserved failure.
            if let Err(error) = known
                .stream
                .retirement_fence()
                .and_then(|receipt| receipt.wait())
            {
                let error = Error::from(error);
                if failure.is_none() {
                    failure = Some(error);
                } else {
                    eprintln!("secondary native retirement cleanup failed: {error}");
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
thread_local! {
    static ACTIVE_STREAMS:RefCell<Option<Rc<NativeRetirementStreams>>>=const{RefCell::new(None)};
}
/// Actual synchronous invocation only. Drop restores selection, never drains or
/// grants credit. Rc makes the guard thread-affine and unsuitable across await.
pub struct NativeRetirementStreamScope {
    previous: Option<Rc<NativeRetirementStreams>>,
}
impl NativeRetirementStreamScope {
    pub fn enter(streams: &Rc<NativeRetirementStreams>) -> Result<Self> {
        streams.validate_current()?;
        Ok(Self {
            previous: ACTIVE_STREAMS.with(|slot| slot.replace(Some(streams.clone()))),
        })
    }
}
impl Drop for NativeRetirementStreamScope {
    fn drop(&mut self) {
        ACTIVE_STREAMS.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}
fn active_streams() -> Result<Rc<NativeRetirementStreams>> {
    ACTIVE_STREAMS
        .with(|slot| slot.borrow().clone())
        .ok_or_else(|| {
            Error::Unsupported("native retirement has no prepared relevant-stream cache".into())
        })
}
pub fn drain_native_retirement() -> Result<()> {
    active_streams()?.drain()
}
pub fn retirement_selected_stream_index() -> Result<i32> {
    Ok(active_streams()?.selected)
}
pub fn validate_native_retirement_streams() -> Result<()> {
    if native_retirement_binding().is_some() {
        active_streams()?.validate_current()?;
    }
    Ok(())
}
/// Execute one ACTUALLY configured synchronous invocation. The caller owns the
/// provider configure step and supplies its exact opaque lease. Every error
/// path drains cached ranges and runs provider cleanup; only complete success
/// publishes the endpoint. Generic E preserves the caller's ORIGINAL typed error.
pub fn invoke_native_retirement<T, E>(
    lease: &NativeRetirementLease,
    streams: &Rc<NativeRetirementStreams>,
    actual: &crate::GenerationRequest,
    body: impl FnOnce() -> std::result::Result<T, E>,
    finish: impl FnOnce(crate::gen_core::MemoryRunOutcome) -> std::result::Result<(), E>,
    canceled: impl Fn(&E) -> bool,
) -> std::result::Result<T, E>
where
    E: From<Error> + std::fmt::Display,
{
    use crate::gen_core::MemoryRunOutcome;
    let activation: Result<_> = (|| {
        let stream_scope = NativeRetirementStreamScope::enter(streams)?;
        let active = NativeRetirementScope::enter_configured(lease)?;
        bind_native_retirement_request(actual as *const crate::GenerationRequest as usize)?;
        Ok((active, stream_scope))
    })();
    let (active, _stream_scope) = match activation {
        Ok(active) => active,
        Err(error) => {
            let primary = E::from(error);
            if let Err(cleanup) = streams.drain() {
                eprintln!("native activation drain failed: {cleanup}");
            }
            if let Err(cleanup) = finish(MemoryRunOutcome::Error {
                message: primary.to_string(),
            }) {
                eprintln!("native activation provider cleanup failed: {cleanup}");
            }
            if let Err(cleanup) = streams.drain() {
                eprintln!("native activation final drain failed: {cleanup}");
            }
            return Err(primary);
        }
    };
    let result = match begin_native_retirement_invocation(actual) {
        Ok(()) => body(),
        Err(error) => Err(E::from(error)),
    };
    let retired = if result.is_ok() {
        capture_native_retirement_completion()
            .map(Some)
            .map_err(E::from)
    } else {
        streams.drain().map(|_| None).map_err(E::from)
    };
    let outcome = match (&result, &retired) {
        (Err(error), _) if canceled(error) => MemoryRunOutcome::Canceled,
        (Err(error), _) | (_, Err(error)) => MemoryRunOutcome::Error {
            message: error.to_string(),
        },
        _ => MemoryRunOutcome::Complete,
    };
    let finished = finish(outcome);
    let final_drain = if result.is_ok() && finished.is_ok() && matches!(&retired, Ok(Some(_))) {
        Ok(())
    } else {
        streams.drain().map_err(E::from)
    };
    let publication = match retired {
        Ok(Some(receipt)) if result.is_ok() && finished.is_ok() => {
            receipt.publish().map_err(E::from)
        }
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    };
    drop(active);
    let mut failure = None;
    for result in [publication, finished, final_drain] {
        if let Err(error) = result {
            if failure.is_none() {
                failure = Some(error);
            } else {
                eprintln!("secondary native lifecycle cleanup failed: {error}");
            }
        }
    }
    match result {
        Err(primary) => {
            if let Some(cleanup) = failure {
                eprintln!("native lifecycle cleanup failed after body error: {cleanup}");
            }
            Err(primary)
        }
        Ok(output) => match failure {
            Some(error) => Err(error),
            None => Ok(output),
        },
    }
}

pub fn begin_native_retirement_invocation(actual: &crate::GenerationRequest) -> Result<()> {
    validate_native_retirement_scope()?;
    if !enabled()
        || native_retirement_binding().is_none()
        || !native_retirement_request_matches(actual as *const crate::GenerationRequest as usize)?
    {
        return Err(Error::Unsupported(
            "admitted retirement invocation requires its actual configured quantum scope".into(),
        ));
    }
    active_streams()?.validate_current()?;
    let epoch = native_retirement_epoch()?
        .ok_or_else(|| Error::Unsupported("native invocation missing".into()))?;
    let binding = native_retirement_binding().expect("validated binding");
    let streams = active_streams()?.observation();
    diagnostic_metadata(
        binding.provider,
        "native-lifecycle",
        &[
            ("schema", 1),
            ("invocation", epoch.invocation),
            ("started", 1),
            ("configured", 1),
            ("epoch", epoch.epoch),
            streams[0],
            streams[1],
            streams[2],
            streams[3],
            streams[4],
            streams[5],
            streams[6],
            ("exclusive_request_ownership", 0),
            ("host_reserve_declared", 0),
        ],
    );
    Ok(())
}
/// Real successful captured range, never fabricated or sent across threads.
/// Drop discards it without publishing a successful invocation endpoint.
pub struct NativeRetirementCompletion {
    invocation: u64,
    provider: &'static str,
    streams: Rc<NativeRetirementStreams>,
}
pub fn capture_native_retirement_completion() -> Result<NativeRetirementCompletion> {
    validate_native_retirement_scope()?;
    let epoch = native_retirement_epoch()?
        .ok_or_else(|| Error::Unsupported("native invocation missing".into()))?;
    let binding = native_retirement_binding()
        .ok_or_else(|| Error::Unsupported("native lifecycle expired".into()))?;
    let streams = active_streams()?;
    streams.drain()?;
    Ok(NativeRetirementCompletion {
        invocation: epoch.invocation,
        provider: binding.provider,
        streams,
    })
}
impl NativeRetirementCompletion {
    /// Only after body AND provider finish succeeded. Drain again AFTER provider
    /// cleanup, using cached handles even though finish invalidated registration.
    pub fn publish(self) -> Result<()> {
        self.streams.drain()?;
        let epoch = super::retirement::retirement_chronology()
            .ok_or_else(|| Error::Unsupported("native selector ended before publication".into()))?;
        if epoch.invocation != self.invocation {
            return Err(Error::Unsupported(
                "native invocation changed before publication".into(),
            ));
        }
        diagnostic_metadata(
            self.provider,
            "native-lifecycle-completed",
            &[
                ("schema", 1),
                ("invocation", self.invocation),
                ("status", 1),
                ("epochs", epoch.epoch),
            ],
        );
        Ok(())
    }
}
pub(super) fn checked_native_retirement(label: &str) -> Result<()> {
    validate_native_retirement_streams()?;
    if let Some(epoch) = begin_native_retirement_epoch()? {
        diagnostic_metadata(
            "mlx",
            "native-quantum",
            &[
                ("schema", 1),
                ("invocation", epoch.invocation),
                ("epoch", epoch.epoch),
                ("label", retirement_label_id(label)),
            ],
        );
    }
    Ok(())
}
pub(super) fn synchronize_native_retirement(label: &str) -> Result<()> {
    validate_native_retirement_scope()?;
    drain_native_retirement()?;
    let epoch = native_retirement_epoch()?
        .ok_or_else(|| Error::Unsupported("native quantum lost its invocation".into()))?;
    let streams = active_streams()?.observation();
    diagnostic_metadata(
        "mlx",
        "native-joined",
        &[
            ("schema", 1),
            ("invocation", epoch.invocation),
            ("epoch", epoch.epoch),
            ("label", retirement_label_id(label)),
            ("all_source_streams", 1),
            ("status", 1),
            streams[0],
            streams[1],
            streams[2],
            streams[3],
            streams[4],
            streams[5],
            streams[6],
            ("range_captured_prefix", 1),
            ("exclusive_request_ownership", 0),
            ("alias_release_credit", 0),
            ("cache_release_credit", 0),
            ("process_release_credit", 0),
            ("caller_receipts_live_upper", 1),
            ("host_reserve_declared", 0),
        ],
    );
    Ok(())
}
