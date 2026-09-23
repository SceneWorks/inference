//! Compute device + dtype selection.
//!
//! Follows the `candle-gen` convention: the backend is chosen at compile time by feature
//! (CUDA → Metal → CPU). The compute dtype is `bf16` on the GPU backends (matching the `mlx-llm`
//! reference) and `f32` on CPU, where half-precision kernels are slow or unsupported.

use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// Environment switch for the CUDA stream the model runs on (story sc-24134): `legacy` or `own`.
/// Unset (the default) selects `own` only when the CUDA-graph runner is switched on at
/// [`select_device`] time and `legacy` otherwise; a `flash-attn` build is always `legacy`. See
/// [`CudaStreamKind::resolve`].
pub const CUDA_STREAM_ENV: &str = "CANDLE_LLM_CUDA_STREAM";

/// Which CUDA stream [`select_device`] puts the model on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStreamKind {
    /// The legacy NULL stream `Device::new_cuda` uses, with cudarc's default event tracking —
    /// the **default**. Every CUDA device a process creates on it shares the legacy stream of
    /// the primary context, so work is ordered whichever device issued it. Stream capture is
    /// not supported on it: the CUDA-graph runner refuses it (`legacy_stream`).
    Legacy,
    /// The model's **own** non-blocking stream (`Device::new_cuda_with_stream`) with cudarc's
    /// per-slice event tracking off — the only form a CUDA graph can be captured on. Selected
    /// when the CUDA-graph runner is switched on at [`select_device`] time, or explicitly with
    /// `CANDLE_LLM_CUDA_STREAM=own`; never in a `flash-attn` build.
    Own,
}

impl CudaStreamKind {
    /// Parse the switch's value (case-insensitive): unset / empty → `None` (the default, see
    /// [`resolve`](Self::resolve)), `own` → [`Own`](Self::Own), `legacy` →
    /// [`Legacy`](Self::Legacy), anything else a configuration error.
    pub fn parse(value: Option<&str>) -> Result<Option<Self>> {
        match value.map(|v| v.trim().to_ascii_lowercase()) {
            None => Ok(None),
            Some(v) if v.is_empty() => Ok(None),
            Some(v) if v == "own" => Ok(Some(Self::Own)),
            Some(v) if v == "legacy" => Ok(Some(Self::Legacy)),
            Some(v) => Err(Error::Config(format!(
                "unsupported {CUDA_STREAM_ENV}={v:?}; expected `own` or `legacy`"
            ))),
        }
    }

    /// The stream a device gets: an explicit `requested` kind wins; unset, the model gets its
    /// own stream only when the CUDA-graph runner is on (`cuda_graphs`) — the one consumer that
    /// needs it — and the legacy stream otherwise. A `flash-attn` build is always
    /// [`Legacy`](Self::Legacy): candle-flash-attn launches its kernels on stream 0, which does
    /// not synchronize with a non-blocking stream, so an own-stream model would race its own
    /// attention.
    pub fn resolve(requested: Option<Self>, cuda_graphs: bool) -> Self {
        if cfg!(feature = "flash-attn") {
            return Self::Legacy;
        }
        match requested {
            Some(kind) => kind,
            None if cuda_graphs => Self::Own,
            None => Self::Legacy,
        }
    }

    /// The stream [`select_device`] would pick now: the environment switch resolved against the
    /// CUDA-graph runner's switch ([`cuda_graphs_enabled`](crate::decode::cuda_graphs_enabled)).
    pub fn from_env() -> Result<Self> {
        let value = std::env::var(CUDA_STREAM_ENV).ok();
        Ok(Self::resolve(
            Self::parse(value.as_deref())?,
            crate::decode::cuda_graphs_enabled(),
        ))
    }

    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::Legacy => "legacy",
        }
    }
}

/// The process-default compute device, selected at compile time by feature:
/// CUDA (`cuda`) → Metal (`metal`) → CPU (default).
///
/// Call it **once** per loaded model and keep the device: on the own stream every call creates
/// a new stream (with its own cuBLAS / cuRAND handles) that candle's op-level device check —
/// which compares the GPU ordinal only — cannot tell apart from the model's.
pub fn select_device() -> Result<Device> {
    if let Some(selection) = std::env::var_os("CANDLE_LLM_DEVICE") {
        let selection = selection.to_string_lossy();
        if selection.eq_ignore_ascii_case("cpu") {
            return Ok(Device::Cpu);
        }
        if !selection.eq_ignore_ascii_case("auto") {
            return Err(crate::error::Error::Config(format!(
                "unsupported CANDLE_LLM_DEVICE={selection:?}; expected `auto` or `cpu`"
            )));
        }
    }
    // The stream (story sc-24134): the legacy NULL stream unless the CUDA-graph runner — which
    // records a decode step with stream capture, unsupported on the legacy stream — is on at
    // this point, or `CANDLE_LLM_CUDA_STREAM=own` asks for it (`CudaStreamKind::resolve`).
    #[cfg(feature = "cuda")]
    let dev = cuda_device(CudaStreamKind::from_env()?)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let dev = Device::Cpu;
    Ok(dev)
}

/// CUDA device 0 on the stream `kind` names.
///
/// On the own stream cudarc's per-slice event tracking is switched off. While it is on, cudarc
/// records two CUDA events per allocation and makes a stream wait on them once a second stream
/// exists, and a wait on an event recorded before a capture invalidates the capture. Switching
/// it off is the `unsafe` contract: nothing orders this stream against any other stream any
/// more, so every tensor the model's kernels touch must be issued through this one device.
/// That holds because each provider calls [`select_device`] once, at load, and hands that
/// device to everything it builds (StarVector's request pixels included), and because a
/// `flash-attn` build — whose kernels launch on stream 0 — never gets the own stream. candle
/// does not enforce it: its per-op device check compares the GPU ordinal only.
#[cfg(feature = "cuda")]
fn cuda_device(kind: CudaStreamKind) -> Result<Device> {
    Ok(match kind {
        CudaStreamKind::Own => {
            let dev = Device::new_cuda_with_stream(0)?;
            if let Device::Cuda(cuda) = &dev {
                // SAFETY: the single-device contract above.
                unsafe { cuda.disable_event_tracking() };
            }
            dev
        }
        CudaStreamKind::Legacy => Device::new_cuda(0)?,
    })
}

/// The dense compute dtype for a device: `bf16` on the GPU backends (CUDA / Metal — matching the
/// mlx-llm reference engine), `f32` on CPU.
pub fn compute_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_stream_switch_parses_own_legacy_and_refuses_the_rest() {
        assert_eq!(CudaStreamKind::parse(None).unwrap(), None);
        assert_eq!(CudaStreamKind::parse(Some(" ")).unwrap(), None);
        assert_eq!(
            CudaStreamKind::parse(Some(" Own ")).unwrap(),
            Some(CudaStreamKind::Own)
        );
        assert_eq!(
            CudaStreamKind::parse(Some("LEGACY")).unwrap(),
            Some(CudaStreamKind::Legacy)
        );
        assert!(matches!(
            CudaStreamKind::parse(Some("null")),
            Err(Error::Config(_))
        ));
        assert_eq!(CudaStreamKind::Own.label(), "own");
        assert_eq!(CudaStreamKind::Legacy.label(), "legacy");
    }

    /// The default is the legacy stream; the own stream only when the graph runner is on at
    /// selection time or it is asked for by name — and never in a `flash-attn` build.
    #[test]
    fn stream_default_is_legacy_unless_graphs_are_on_or_own_is_asked_for() {
        use CudaStreamKind::{Legacy, Own};
        let own_unless_flash = if cfg!(feature = "flash-attn") {
            Legacy
        } else {
            Own
        };
        assert_eq!(CudaStreamKind::resolve(None, false), Legacy);
        assert_eq!(CudaStreamKind::resolve(None, true), own_unless_flash);
        assert_eq!(CudaStreamKind::resolve(Some(Own), false), own_unless_flash);
        assert_eq!(CudaStreamKind::resolve(Some(Legacy), true), Legacy);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::decode::graph::cuda_graphs_policy_guard;

    /// `select_device` puts the model on the legacy NULL stream (cudarc event tracking on) with
    /// the graph runner off, and on its own stream with tracking off when the runner is on at
    /// selection time (a `flash-attn` build stays legacy).
    #[test]
    fn select_device_is_legacy_by_default_and_own_when_the_graph_runner_is_on() {
        if std::env::var_os(CUDA_STREAM_ENV).is_some() {
            eprintln!("skipping: {CUDA_STREAM_ENV} is set");
            return;
        }
        // (null stream, event tracking) of the device `select_device` returns.
        let selected = |graphs_on: bool| {
            let _guard = cuda_graphs_policy_guard(Some(graphs_on));
            match select_device() {
                Ok(Device::Cuda(d)) => {
                    Some((d.cuda_stream().cu_stream().is_null(), d.is_event_tracking()))
                }
                _ => None,
            }
        };
        let Some(off) = selected(false) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        assert_eq!(
            off,
            (true, true),
            "graphs off: the legacy stream, tracking on"
        );
        let expected_on = if cfg!(feature = "flash-attn") {
            (true, true)
        } else {
            (false, false)
        };
        assert_eq!(
            selected(true),
            Some(expected_on),
            "graphs on: the own stream, tracking off"
        );
    }
}
