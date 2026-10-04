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
pub(crate) fn stream_env_value() -> Option<&'static str> {
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    VALUE
        .get_or_init(|| std::env::var(CUDA_STREAM_ENV).ok())
        .as_deref()
}

/// Which CUDA stream [`select_device`] puts the model on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStreamKind {
    /// The legacy NULL stream `Device::new_cuda` uses, with cudarc's default event tracking —
    /// the stream whenever the CUDA-graph runner is off at [`select_device`] time, in every
    /// `flash-attn` build, for every device a provider that never captures opens
    /// ([`select_eager_device`]), or with `CANDLE_LLM_CUDA_STREAM=legacy`. Every CUDA device a
    /// process creates on it shares the legacy stream of the primary context, so work is ordered
    /// whichever device issued it. Stream capture is not supported on it: the CUDA-graph runner
    /// refuses it (`legacy_stream`).
    Legacy,
    /// The model's **own** non-blocking stream (`Device::new_cuda_with_stream`) with cudarc's
    /// per-slice event tracking off — the only form a CUDA graph can be captured on. Selected
    /// when the CUDA-graph runner is on at [`select_device`] time — the CUDA default since
    /// sc-24446 (`core_llm::defaults::CANDLE_CUDA.cuda_graphs`) — or explicitly with
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

    /// The stream [`select_eager_device`] picks: [`from_env`](Self::from_env) with the graph
    /// runner off on this thread — legacy unless `CANDLE_LLM_CUDA_STREAM=own` asks otherwise.
    pub fn eager_from_env() -> Result<Self> {
        let _eager = crate::decode::cuda_graphs_scope(Some(false));
        Self::from_env()
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
    select_device_on(CudaStreamKind::from_env)
}

/// [`select_device`] for a provider that never runs the CUDA-graph runner (LLaVA, StarVector,
/// the capabilities probe): its CUDA device stays on the legacy stream
/// ([`CudaStreamKind::eager_from_env`]) whatever the graph switch says, since an own stream only
/// serves a capture and drops cudarc's cross-stream ordering. Same call-once rule.
pub fn select_eager_device() -> Result<Device> {
    select_device_on(CudaStreamKind::eager_from_env)
}

fn select_device_on(stream: fn() -> Result<CudaStreamKind>) -> Result<Device> {
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
    let dev = cuda_device(stream()?)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let dev = Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let dev = Device::Cpu;
    #[cfg(not(feature = "cuda"))]
    let _ = stream;
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

/// The decode-defaults row ([`core_llm::defaults`], epic sc-24432 E5) a model on `device` runs
/// with: Candle CUDA, Candle Metal or Candle CPU.
pub fn decode_backend(device: &Device) -> core_llm::DecodeBackend {
    if device.is_cuda() {
        core_llm::DecodeBackend::CandleCuda
    } else if device.is_metal() {
        core_llm::DecodeBackend::CandleMetal
    } else {
        core_llm::DecodeBackend::CandleCpu
    }
}

/// Whether [`select_device`] would open a CUDA device in this process — a `cuda` build unless
/// `CANDLE_LLM_DEVICE=cpu` — **without opening it** (a second CUDA device would put a second
/// stream on the context).
pub(crate) fn selected_device_is_cuda() -> bool {
    cfg!(feature = "cuda") && !cpu_forced()
}

fn cpu_forced() -> bool {
    std::env::var_os("CANDLE_LLM_DEVICE")
        .is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case("cpu"))
}

/// The decode-defaults row of the device [`select_device`] opens, from whether it is CUDA alone
/// (a load estimate knows no more): CUDA, else Metal in a `metal` build that `CANDLE_LLM_DEVICE`
/// does not force onto the CPU, else the CPU.
pub(crate) fn decode_backend_for(cuda: bool) -> core_llm::DecodeBackend {
    if cuda {
        core_llm::DecodeBackend::CandleCuda
    } else if cfg!(feature = "metal") && !cpu_forced() {
        core_llm::DecodeBackend::CandleMetal
    } else {
        core_llm::DecodeBackend::CandleCpu
    }
}

/// [`decode_backend`]'s row of the defaults table.
pub fn decode_defaults(device: &Device) -> &'static core_llm::DecodeDefaults {
    decode_backend(device).defaults()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-24446 (E5): a device maps onto its row of the defaults table, and the process switches
    /// whose path exists only on CUDA (fused primitives, NVFP4 GEMV) take their unset state from
    /// the Candle CUDA row; the CUDA-graph switch takes the device's row (CUDA's on CUDA).
    #[test]
    fn devices_map_onto_their_defaults_row_and_the_cuda_switches_read_the_cuda_row() {
        use core_llm::defaults::CANDLE_CUDA;
        use core_llm::DecodeBackend;
        assert_eq!(decode_backend(&Device::Cpu), DecodeBackend::CandleCpu);
        assert_eq!(
            decode_defaults(&Device::Cpu).backend,
            DecodeBackend::CandleCpu
        );
        assert_eq!(decode_backend_for(true), DecodeBackend::CandleCuda);
        let unset = |env: &str| std::env::var_os(env).is_none();
        if unset(crate::primitives::fused::FUSED_KERNELS_ENV) {
            let _policy = crate::primitives::fused::fused_policy_guard(None);
            assert_eq!(
                crate::primitives::fused::fused_kernels_enabled(),
                CANDLE_CUDA.fused_kernels
            );
        }
        if unset(crate::primitives::nvfp4_path::NVFP4_GEMV_ENV) {
            let _policy = crate::primitives::nvfp4_path::nvfp4_gemv_policy_guard(None);
            assert_eq!(
                crate::primitives::nvfp4_path::nvfp4_gemv_enabled(),
                CANDLE_CUDA.nvfp4_gemv
            );
        }
        assert_eq!(
            crate::decode::graph::CUDA_GRAPHS_DEFAULT,
            CANDLE_CUDA.cuda_graphs
        );
    }

    /// sc-24446: a provider that never captures (LLaVA, StarVector-1B / -8B, the capabilities
    /// probe) opens its device through `select_eager_device`, whose stream resolves legacy with
    /// the graph switch on — where `select_device` resolves own (outside a `flash-attn` build).
    #[test]
    fn providers_that_never_capture_resolve_the_legacy_stream() {
        let _graphs = crate::decode::graph::cuda_graphs_policy_guard(Some(true));
        if stream_env_value().is_none() {
            assert_eq!(
                CudaStreamKind::eager_from_env().unwrap(),
                CudaStreamKind::Legacy
            );
            let graph_stream = if cfg!(feature = "flash-attn") {
                CudaStreamKind::Legacy
            } else {
                CudaStreamKind::Own
            };
            assert_eq!(CudaStreamKind::from_env().unwrap(), graph_stream);
        }
        for (name, source) in [
            ("llava.rs", include_str!("llava.rs")),
            ("starvector.rs", include_str!("starvector.rs")),
            ("starvector_8b.rs", include_str!("starvector_8b.rs")),
            ("backend.rs", include_str!("backend.rs")),
        ] {
            let production = source.split("mod tests {").next().unwrap();
            assert!(production.contains("select_eager_device()"), "{name}");
            assert!(!production.contains("select_device()"), "{name}");
        }
    }

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

    /// sc-24446: `select_eager_device` opens the legacy NULL stream (tracking on) even with the
    /// graph runner on.
    #[test]
    fn select_eager_device_stays_on_the_legacy_stream_with_graphs_on() {
        if std::env::var_os(CUDA_STREAM_ENV).is_some() {
            eprintln!("skipping: {CUDA_STREAM_ENV} is set");
            return;
        }
        let _guard = cuda_graphs_policy_guard(Some(true));
        match select_eager_device() {
            Ok(Device::Cuda(d)) => assert_eq!(
                (d.cuda_stream().cu_stream().is_null(), d.is_event_tracking()),
                (true, true)
            ),
            _ => eprintln!("skipping: no CUDA device"),
        }
    }

    /// sc-24140 feature-end review: a test thread that opens a CUDA device — through
    /// `select_device` or `new_cuda_for_test` — holds the CUDA test lock (the graph guard's) until
    /// it exits, so no other test's capture can overlap its legacy-stream launches; the lock is
    /// released when the thread ends. Every wait is bounded.
    #[test]
    fn opening_a_cuda_device_holds_the_cuda_test_lock_until_the_thread_ends() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        type Open = fn() -> Result<Device>;
        let openers: [(&str, Open); 2] = [
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
            let taken_elsewhere =
                std::thread::spawn(|| crate::decode::graph::try_cuda_test_lock().is_some())
                    .join()
                    .unwrap();
            assert!(
                !taken_elsewhere,
                "{name}: no other thread can take the CUDA test lock while a test thread holds a \
                 CUDA device"
            );
            let (guarded_tx, guarded) = channel();
            let waiter = std::thread::spawn(move || {
                let _guard = cuda_graphs_policy_guard(None);
                guarded_tx.send(()).unwrap();
            });
            finish_tx.send(()).unwrap();
            holder.join().unwrap();
            guarded
                .recv_timeout(Duration::from_secs(600))
                .unwrap_or_else(|_| panic!("{name}: the lock is released when the thread ends"));
            waiter.join().unwrap();
        }
    }
}
