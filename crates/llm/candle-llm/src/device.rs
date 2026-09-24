//! Compute device + dtype selection.
//!
//! Follows the `candle-gen` convention: the backend is chosen at compile time by feature
//! (CUDA → Metal → CPU). The compute dtype is `bf16` on the GPU backends (matching the `mlx-llm`
//! reference) and `f32` on CPU, where half-precision kernels are slow or unsupported.

use std::sync::OnceLock;

use candle_core::{DType, Device};

use crate::error::{Error, Result};

/// Environment switch for the CUDA stream the model runs on (story sc-24134): `legacy` or `own`.
/// Unset (the default) selects `own` only when the CUDA-graph runner is switched on at
/// [`select_device`] time and `legacy` otherwise; a `flash-attn` build is always `legacy`. See
/// [`CudaStreamKind::resolve`].
pub const CUDA_STREAM_ENV: &str = "CANDLE_LLM_CUDA_STREAM";

/// [`CUDA_STREAM_ENV`]'s value, read once per process and cached.
fn stream_env_value() -> Option<&'static str> {
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    VALUE
        .get_or_init(|| std::env::var(CUDA_STREAM_ENV).ok())
        .as_deref()
}

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
    /// The variable is read once per process, like every other switch (sc-24140).
    pub fn from_env() -> Result<Self> {
        Ok(Self::resolve(
            Self::parse(stream_env_value())?,
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
    // A unit test that opens a CUDA device runs behind the CUDA test lock until it ends
    // (sc-24140 feature-end review; `crate::decode::graph::hold_cuda_test_lock`).
    #[cfg(all(test, feature = "cuda"))]
    crate::decode::graph::hold_cuda_test_lock();
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

/// Unit-test seam (sc-24140 feature-end review): `Device::new_cuda(0)` — a device on the legacy
/// NULL stream — taken behind the CUDA test lock
/// ([`hold_cuda_test_lock`](crate::decode::graph::hold_cuda_test_lock)) for the rest of the
/// calling test, so its launches never overlap another test's stream capture. Every unit test
/// opens a CUDA device through this or [`select_device`], never `Device::new_cuda` directly.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn new_cuda_for_test() -> Result<Device> {
    crate::decode::graph::hold_cuda_test_lock();
    Ok(Device::new_cuda(0)?)
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

    /// The stream switch is read once per process like every other switch (sc-24140): a later
    /// change of the variable is not seen.
    #[test]
    fn the_stream_switch_is_read_once_per_process() {
        let first = stream_env_value().map(str::to_owned);
        let was = std::env::var_os(CUDA_STREAM_ENV);
        std::env::set_var(CUDA_STREAM_ENV, "not-a-stream-kind");
        let second = stream_env_value().map(str::to_owned);
        match was {
            Some(value) => std::env::set_var(CUDA_STREAM_ENV, value),
            None => std::env::remove_var(CUDA_STREAM_ENV),
        }
        assert_eq!(second, first, "the cached value, not a re-read");
    }

    /// sc-24140 feature-end review: a unit test opens a CUDA device only through `select_device`
    /// or `new_cuda_for_test`, both behind the CUDA test lock. A direct `Device::new_cuda` (or
    /// `new_cuda_with_stream`) anywhere else in `src/` could launch on the legacy stream while
    /// another test captures a graph — the `capture_invalidated` flake. The only direct calls are
    /// `cuda_device`'s two and the test helper's one.
    #[test]
    fn cuda_devices_are_opened_only_behind_the_cuda_test_lock() {
        fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    rust_files(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        // Split so this test's own source is not a match.
        let needles = [
            ["Device::new_", "cuda("].concat(),
            ["new_cuda_", "with_stream("].concat(),
        ];
        let mut direct = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let name = file
                .strip_prefix(&src)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            for (number, line) in text.lines().enumerate() {
                let code = line.trim_start();
                if !code.starts_with("//") && needles.iter().any(|n| code.contains(n.as_str())) {
                    direct.push(format!("{name}:{}", number + 1));
                }
            }
        }
        assert_eq!(direct.len(), 3, "{direct:?}");
        assert!(
            direct.iter().all(|site| site.starts_with("device.rs:")),
            "open CUDA devices in tests through `new_cuda_for_test` or `select_device`: {direct:?}"
        );
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

    /// sc-24140 feature-end review: a test thread that opens a CUDA device — through
    /// `select_device` or `new_cuda_for_test` — holds the CUDA test lock (the graph guard's) until
    /// it exits, so no other test's capture can overlap its legacy-stream launches; the lock is
    /// released when the thread ends. Every wait is bounded.
    #[test]
    fn opening_a_cuda_device_holds_the_cuda_test_lock_until_the_thread_ends() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let openers: [(&str, fn() -> Result<Device>); 2] = [
            ("select_device", select_device),
            ("new_cuda_for_test", new_cuda_for_test),
        ];
        for (name, open) in openers {
            let (opened_tx, opened) = channel();
            let (finish_tx, finish) = channel::<()>();
            let holder = std::thread::spawn(move || {
                opened_tx.send(open().is_ok_and(|d| d.is_cuda())).unwrap();
                finish.recv().unwrap();
            });
            if !opened.recv_timeout(Duration::from_secs(600)).unwrap() {
                eprintln!("skipping: no CUDA device");
                finish_tx.send(()).unwrap();
                holder.join().unwrap();
                return;
            }
            let (guarded_tx, guarded) = channel();
            let waiter = std::thread::spawn(move || {
                let _guard = cuda_graphs_policy_guard(None);
                guarded_tx.send(()).unwrap();
            });
            assert!(
                guarded.recv_timeout(Duration::from_millis(500)).is_err(),
                "{name}: a capture guard must wait while a test thread holds a CUDA device"
            );
            finish_tx.send(()).unwrap();
            holder.join().unwrap();
            guarded
                .recv_timeout(Duration::from_secs(600))
                .unwrap_or_else(|_| panic!("{name}: the lock is released when the thread ends"));
            waiter.join().unwrap();
        }
    }
}
