//! The shared nvrtc **compile-once** seam (sc-23990, landed by sc-24137 for epic sc-24128).
//!
//! Every runtime-compiled CUDA kernel in the workspace goes through one mechanism: a
//! [`KernelSource`] names an `include_str!`'d `.cu` source and the compute-capability floor it
//! needs, and [`KernelSource::compiled`] turns it into a loaded module for one device. There is no
//! `build.rs` and no `nvcc` step — nvrtc JITs the source for the live device at first use.
//!
//! The cache is **per device, per kernel** (keyed by the CUDA device ordinal and the kernel name)
//! and it stores the *outcome*: a successful compile is held for the life of the process, and a
//! **failure is cached too**, so a kernel nvrtc cannot build on this host is refused with the same
//! typed [`KernelCompileError`] on every later call without recompiling (epic E3). The fused
//! decode primitives fall back to their op-chain reference on that error, and the error is what
//! their telemetry reports as the reason.
//!
//! Compilation targets the device's own architecture (`--gpu-architecture=compute_XY`, the
//! highest known virtual architecture at or below the device's capability), so PTX instructions
//! gated on a minimum SM — the bf16 conversions the fused decode kernels use — assemble; below
//! the declared floor the source is refused before nvrtc runs.
//!
//! Device code lives behind `cfg(feature = "cuda")`; the source descriptor and the error type
//! build everywhere so CPU lanes can name kernels and match on their errors.

use std::fmt;

/// A runtime-compiled CUDA kernel source: the `.cu` text (`include_str!`), a stable name (the
/// cache key, unique per source) and the compute-capability floor the code needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSource {
    /// Cache key and diagnostic name; unique per distinct source in the process.
    pub name: &'static str,
    /// The CUDA C++ source, compiled as-is (no include paths: sources use only builtins).
    pub src: &'static str,
    /// `(major, minor)` compute-capability floor; devices below it are refused without compiling.
    pub cc_floor: (i32, i32),
}

/// Why a [`KernelSource`] is not usable on a device. Cached per `(device, kernel)` and returned
/// verbatim on every later call, so matching on the variant is stable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelCompileError {
    /// The device's compute capability is below the source's declared floor.
    BelowComputeFloor {
        /// Kernel name.
        name: &'static str,
        /// Declared floor.
        floor: (i32, i32),
        /// The device's capability.
        found: (i32, i32),
    },
    /// nvrtc rejected the source (the message carries the compiler log).
    Nvrtc {
        /// Kernel name.
        name: &'static str,
        /// nvrtc's diagnostic.
        message: String,
    },
    /// The PTX compiled but the driver refused to load the module.
    Load {
        /// Kernel name.
        name: &'static str,
        /// Driver diagnostic.
        message: String,
    },
    /// Querying the device (its capability) failed.
    Device {
        /// Kernel name.
        name: &'static str,
        /// Driver diagnostic.
        message: String,
    },
    /// The module loaded but has no function of the requested name.
    MissingFunction {
        /// Kernel name.
        name: &'static str,
        /// The `extern "C" __global__` symbol that was asked for.
        function: String,
    },
}

impl KernelCompileError {
    /// The kernel this error is about.
    pub fn kernel(&self) -> &'static str {
        match self {
            Self::BelowComputeFloor { name, .. }
            | Self::Nvrtc { name, .. }
            | Self::Load { name, .. }
            | Self::Device { name, .. }
            | Self::MissingFunction { name, .. } => name,
        }
    }

    /// A short stable label for telemetry (`compute_floor`, `nvrtc`, `load`, `device`,
    /// `missing_function`).
    pub fn label(&self) -> &'static str {
        match self {
            Self::BelowComputeFloor { .. } => "compute_floor",
            Self::Nvrtc { .. } => "nvrtc",
            Self::Load { .. } => "load",
            Self::Device { .. } => "device",
            Self::MissingFunction { .. } => "missing_function",
        }
    }
}

impl fmt::Display for KernelCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BelowComputeFloor { name, floor, found } => write!(
                f,
                "kernel `{name}` needs compute capability >= sm_{}{} (device is sm_{}{})",
                floor.0, floor.1, found.0, found.1
            ),
            Self::Nvrtc { name, message } => {
                write!(f, "nvrtc failed to compile kernel `{name}`: {message}")
            }
            Self::Load { name, message } => {
                write!(
                    f,
                    "the driver refused the PTX of kernel `{name}`: {message}"
                )
            }
            Self::Device { name, message } => {
                write!(
                    f,
                    "querying the device for kernel `{name}` failed: {message}"
                )
            }
            Self::MissingFunction { name, function } => {
                write!(f, "kernel `{name}` has no function `{function}`")
            }
        }
    }
}

impl std::error::Error for KernelCompileError {}

/// The highest known nvrtc virtual architecture at or below `cap`, as a `--gpu-architecture`
/// value; `None` below sm_70 (nvrtc's own default then applies).
pub fn nvrtc_arch_for(cap: (i32, i32)) -> Option<&'static str> {
    const KNOWN: &[((i32, i32), &str)] = &[
        ((12, 1), "compute_121"),
        ((12, 0), "compute_120"),
        ((10, 3), "compute_103"),
        ((10, 1), "compute_101"),
        ((10, 0), "compute_100"),
        ((9, 0), "compute_90"),
        ((8, 9), "compute_89"),
        ((8, 7), "compute_87"),
        ((8, 6), "compute_86"),
        ((8, 0), "compute_80"),
        ((7, 5), "compute_75"),
        ((7, 0), "compute_70"),
    ];
    KNOWN
        .iter()
        .find(|(known, _)| *known <= cap)
        .map(|(_, arch)| *arch)
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::CudaDevice;
    use cudarc::driver::{CudaFunction, CudaModule};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    /// A [`KernelSource`] compiled and loaded for one device. Holds the driver module alive; the
    /// functions it hands out keep it alive too.
    pub struct CompiledKernel {
        name: &'static str,
        ordinal: usize,
        compute_cap: (i32, i32),
        module: Arc<CudaModule>,
    }

    impl fmt::Debug for CompiledKernel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CompiledKernel")
                .field("name", &self.name)
                .field("ordinal", &self.ordinal)
                .field("compute_cap", &self.compute_cap)
                .finish()
        }
    }

    impl CompiledKernel {
        /// The `extern "C" __global__` function `function` of this module.
        pub fn function(&self, function: &str) -> Result<CudaFunction, KernelCompileError> {
            self.module
                .load_function(function)
                .map_err(|_| KernelCompileError::MissingFunction {
                    name: self.name,
                    function: function.to_string(),
                })
        }

        /// Kernel name (the source's `name`).
        pub fn name(&self) -> &'static str {
            self.name
        }

        /// CUDA device ordinal this module was loaded on.
        pub fn ordinal(&self) -> usize {
            self.ordinal
        }

        /// The device's compute capability at compile time.
        pub fn compute_cap(&self) -> (i32, i32) {
            self.compute_cap
        }
    }

    type Key = (usize, &'static str);
    type Outcome = Result<Arc<CompiledKernel>, KernelCompileError>;

    #[derive(Default)]
    struct Cache {
        outcomes: HashMap<Key, Outcome>,
        attempts: HashMap<Key, u64>,
    }

    fn cache() -> &'static Mutex<Cache> {
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(Cache::default()))
    }

    fn lock() -> std::sync::MutexGuard<'static, Cache> {
        cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `(major, minor)` compute capability of `dev`.
    pub fn device_compute_cap(dev: &CudaDevice) -> Result<(i32, i32), String> {
        use cudarc::driver::sys::CUdevice_attribute as A;
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let major = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("{e:?}"))?;
        let minor = ctx
            .attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("{e:?}"))?;
        Ok((major, minor))
    }

    impl KernelSource {
        fn key(&self, dev: &CudaDevice) -> Key {
            (dev.cuda_stream().context().ordinal(), self.name)
        }

        /// The compiled module for `dev`: compiled and loaded on the first call per device,
        /// served from the cache after that — **including a failure**, which is returned as the
        /// same [`KernelCompileError`] without touching nvrtc again.
        pub fn compiled(&self, dev: &CudaDevice) -> Outcome {
            let key = self.key(dev);
            let mut guard = lock();
            if let Some(outcome) = guard.outcomes.get(&key) {
                return outcome.clone();
            }
            *guard.attempts.entry(key).or_insert(0) += 1;
            let outcome = self.compile_uncached(dev, key.0);
            guard.outcomes.insert(key, outcome.clone());
            outcome
        }

        fn compile_uncached(&self, dev: &CudaDevice, ordinal: usize) -> Outcome {
            let name = self.name;
            let found = device_compute_cap(dev)
                .map_err(|message| KernelCompileError::Device { name, message })?;
            if found < self.cc_floor {
                return Err(KernelCompileError::BelowComputeFloor {
                    name,
                    floor: self.cc_floor,
                    found,
                });
            }
            let opts = cudarc::nvrtc::CompileOptions {
                arch: nvrtc_arch_for(found),
                name: Some(format!("{name}.cu")),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(self.src, opts).map_err(|e| {
                KernelCompileError::Nvrtc {
                    name,
                    message: format!("{e}"),
                }
            })?;
            let module = dev.cuda_stream().context().load_module(ptx).map_err(|e| {
                KernelCompileError::Load {
                    name,
                    message: format!("{e:?}"),
                }
            })?;
            Ok(Arc::new(CompiledKernel {
                name,
                ordinal,
                compute_cap: found,
                module,
            }))
        }

        /// How many times nvrtc was actually invoked for this source on `dev` (0 before the first
        /// [`compiled`](Self::compiled) call, 1 after it, and still 1 after a cached failure).
        pub fn compile_attempts(&self, dev: &CudaDevice) -> u64 {
            lock().attempts.get(&self.key(dev)).copied().unwrap_or(0)
        }

        /// The cached outcome for `dev` if there is one, without compiling.
        pub fn cached(&self, dev: &CudaDevice) -> Option<Outcome> {
            lock().outcomes.get(&self.key(dev)).cloned()
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{device_compute_cap, CompiledKernel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_picks_the_highest_known_at_or_below_the_device() {
        assert_eq!(nvrtc_arch_for((12, 0)), Some("compute_120"));
        assert_eq!(nvrtc_arch_for((12, 1)), Some("compute_121"));
        assert_eq!(nvrtc_arch_for((12, 5)), Some("compute_121"));
        assert_eq!(nvrtc_arch_for((9, 0)), Some("compute_90"));
        assert_eq!(nvrtc_arch_for((8, 8)), Some("compute_87"));
        assert_eq!(nvrtc_arch_for((6, 1)), None);
    }

    #[test]
    fn errors_name_their_kernel_and_label() {
        let e = KernelCompileError::BelowComputeFloor {
            name: "k",
            floor: (8, 0),
            found: (7, 5),
        };
        assert_eq!(e.kernel(), "k");
        assert_eq!(e.label(), "compute_floor");
        assert!(e.to_string().contains("sm_80"), "{e}");
        let e = KernelCompileError::Nvrtc {
            name: "k",
            message: "boom".into(),
        };
        assert_eq!(e.label(), "nvrtc");
        assert!(e.to_string().contains("boom"));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::Device;
    use std::sync::Arc;

    fn device() -> Option<candle_core::CudaDevice> {
        match Device::new_cuda(0).ok()? {
            Device::Cuda(d) => Some(d),
            _ => None,
        }
    }

    const GOOD: KernelSource = KernelSource {
        name: "sc24137_seam_test_good",
        src: "extern \"C\" __global__ void fill(float* out, int n, float v) { \
              int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = v; }",
        cc_floor: (7, 0),
    };

    const BROKEN: KernelSource = KernelSource {
        name: "sc24137_seam_test_broken",
        src: "this is not CUDA C++ at all;",
        cc_floor: (7, 0),
    };

    const TOO_NEW: KernelSource = KernelSource {
        name: "sc24137_seam_test_too_new",
        src: "extern \"C\" __global__ void k() {}",
        cc_floor: (99, 0),
    };

    #[test]
    fn a_source_is_compiled_once_per_device_and_the_module_is_shared() {
        let Some(dev) = device() else { return };
        assert_eq!(GOOD.compile_attempts(&dev), 0);
        let first = GOOD.compiled(&dev).expect("compiles");
        let second = GOOD.compiled(&dev).expect("cached");
        assert!(
            Arc::ptr_eq(&first, &second),
            "second call must reuse the module"
        );
        assert_eq!(GOOD.compile_attempts(&dev), 1, "nvrtc ran exactly once");
        assert_eq!(first.ordinal(), dev.cuda_stream().context().ordinal());
        assert!(first.compute_cap() >= (7, 0));
        assert_eq!(first.name(), GOOD.name);
        first.function("fill").expect("kernel symbol");
        match first.function("missing") {
            Err(KernelCompileError::MissingFunction { name, function }) => {
                assert_eq!(name, GOOD.name);
                assert_eq!(function, "missing");
            }
            other => panic!("{other:?}"),
        }
        // A second `CudaDevice` handle for the same ordinal hits the same cache entry.
        let again = device().unwrap();
        let third = GOOD.compiled(&again).expect("cached across handles");
        assert!(Arc::ptr_eq(&first, &third));
        assert_eq!(GOOD.compile_attempts(&again), 1);
    }

    #[test]
    fn a_compile_failure_is_cached_and_returned_without_recompiling() {
        let Some(dev) = device() else { return };
        assert!(BROKEN.cached(&dev).is_none());
        let first = BROKEN.compiled(&dev).expect_err("cannot compile");
        assert!(matches!(first, KernelCompileError::Nvrtc { .. }), "{first}");
        assert_eq!(first.kernel(), BROKEN.name);
        assert_eq!(first.label(), "nvrtc");
        let second = BROKEN.compiled(&dev).expect_err("still cannot");
        assert_eq!(first, second, "the cached error is returned verbatim");
        assert_eq!(
            BROKEN.compile_attempts(&dev),
            1,
            "no recompile on the second call"
        );
        assert_eq!(
            BROKEN.cached(&dev).and_then(|outcome| outcome.err()),
            Some(first)
        );
    }

    #[test]
    fn below_the_declared_floor_is_refused_before_nvrtc_runs() {
        let Some(dev) = device() else { return };
        let err = TOO_NEW.compiled(&dev).expect_err("floor");
        match &err {
            KernelCompileError::BelowComputeFloor { name, floor, found } => {
                assert_eq!(*name, TOO_NEW.name);
                assert_eq!(*floor, (99, 0));
                assert_eq!(*found, device_compute_cap(&dev).unwrap());
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            TOO_NEW.compiled(&dev).expect_err("cached floor refusal"),
            err
        );
        assert_eq!(TOO_NEW.compile_attempts(&dev), 1);
    }
}
