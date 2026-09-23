//! The shared nvrtc **compile-once** seam (sc-23990, landed by sc-24137 for epic sc-24128).
//!
//! Every runtime-compiled CUDA kernel in the workspace goes through one mechanism: a
//! [`KernelSource`] names an `include_str!`'d `.cu` source and the compute-capability floor it
//! needs, and `KernelSource::compiled` (cuda builds) turns it into a loaded module for one
//! device. There is no `build.rs` and no `nvcc` step — nvrtc JITs the source for the live device
//! at first use.
//!
//! The cache is **per device, per kernel** (keyed by the CUDA device ordinal and the kernel name)
//! and it stores the *outcome*: a successful compile is held for the life of the process, and a
//! **failure is cached too**, so a kernel nvrtc cannot build on this host is refused with the same
//! typed [`KernelCompileError`] on every later call without recompiling (epic E3). The fused
//! decode primitives fall back to their op-chain reference on that error, and the error is what
//! their telemetry reports as the reason.
//!
//! The process-wide map lock only guards slot lookup: each key owns a `OnceLock` slot, so nvrtc
//! and the module load run outside the map lock (other kernels and other devices are never held
//! up by a compile), while callers racing on the *same* key still block on its slot and nvrtc
//! runs once.
//!
//! A name is bound to its source text on first use: a second, *different* source reusing a name
//! is a programming error and panics on use (naming both), rather than silently being served the
//! first source's module. (Text, not address: a `const` descriptor's `&'static str` is not
//! guaranteed one address across codegen units, so the check compares contents when the
//! addresses differ.)
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
    /// Cache key and diagnostic name; must be unique per distinct source in the process (a reuse
    /// with different text panics on use).
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
    /// One key's outcome, initialised (compiled) outside the map lock.
    type Slot = Arc<OnceLock<Outcome>>;

    #[derive(Default)]
    struct Cache {
        /// Each key's slot and the source text it was first requested with.
        slots: HashMap<Key, (&'static str, Slot)>,
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
        ///
        /// Only the slot lookup holds the process-wide lock; the compile itself runs under this
        /// key's own `OnceLock`, so concurrent callers of the same key wait for the one compile and
        /// everything else proceeds.
        pub fn compiled(&self, dev: &CudaDevice) -> Outcome {
            let key = self.key(dev);
            let (bound, slot) = {
                let mut guard = lock();
                let (bound, slot) = guard
                    .slots
                    .entry(key)
                    .or_insert_with(|| (self.src, Slot::default()));
                (*bound, slot.clone())
            };
            self.assert_same_source(bound);
            slot.get_or_init(|| {
                *lock().attempts.entry(key).or_insert(0) += 1;
                self.compile_uncached(dev, key.0)
            })
            .clone()
        }

        /// Panics if `bound` (the text this name was first compiled from) is a different source.
        fn assert_same_source(&self, bound: &'static str) {
            assert!(
                std::ptr::eq(bound, self.src) || bound == self.src,
                "nvrtc seam: kernel name `{name}` is already bound to a different source; \
                 every KernelSource needs a unique name",
                name = self.name
            );
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
            let (bound, slot) = lock().slots.get(&self.key(dev)).cloned()?;
            self.assert_same_source(bound);
            slot.get().cloned()
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

    const NAMESAKE_A: KernelSource = KernelSource {
        name: "sc24137_seam_test_namesake",
        src: "extern \"C\" __global__ void a(float* out) { out[0] = 1.0f; }",
        cc_floor: (7, 0),
    };

    /// Same name as [`NAMESAKE_A`], different text and a different symbol.
    const NAMESAKE_B: KernelSource = KernelSource {
        name: "sc24137_seam_test_namesake",
        src: "extern \"C\" __global__ void b(float* out) { out[0] = 2.0f; }",
        cc_floor: (7, 0),
    };

    #[test]
    fn a_second_source_reusing_a_name_is_refused_not_served_the_first_module() {
        let Some(dev) = device() else { return };
        let a = NAMESAKE_A.compiled(&dev).expect("A compiles");
        a.function("a").expect("A's own symbol");
        // Without the check this hands back A's module (which has no symbol `b`).
        let refused =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| NAMESAKE_B.compiled(&dev)))
                .expect_err("a different source under a bound name must not be served");
        let message = refused
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            message.contains("sc24137_seam_test_namesake")
                && message.contains("already bound to a different source"),
            "{message}"
        );
        // The first source is unaffected.
        assert!(Arc::ptr_eq(&a, &NAMESAKE_A.compiled(&dev).unwrap()));
        assert_eq!(NAMESAKE_A.compile_attempts(&dev), 1);
    }

    const PER_DEVICE: KernelSource = KernelSource {
        name: "sc24137_seam_test_per_device",
        src: "extern \"C\" __global__ void one(float* out) { out[0] = 1.0f; }",
        cc_floor: (7, 0),
    };

    /// Two real ordinals get two modules, each compiled once. Skips unless a second device is
    /// visible (`CUDA_VISIBLE_DEVICES` restricted to one GPU hides ordinal 1).
    #[test]
    fn each_ordinal_gets_its_own_module_compiled_once() {
        let (Some(dev0), Ok(Device::Cuda(dev1))) = (device(), Device::new_cuda(1)) else {
            eprintln!("skipping: no second CUDA device");
            return;
        };
        let m0 = PER_DEVICE.compiled(&dev0).expect("ordinal 0");
        let m1 = PER_DEVICE.compiled(&dev1).expect("ordinal 1");
        assert!(!Arc::ptr_eq(&m0, &m1), "one module per ordinal");
        assert_eq!(m0.ordinal(), 0);
        assert_eq!(m1.ordinal(), 1);
        assert!(Arc::ptr_eq(&m0, &PER_DEVICE.compiled(&dev0).unwrap()));
        assert!(Arc::ptr_eq(&m1, &PER_DEVICE.compiled(&dev1).unwrap()));
        assert_eq!(PER_DEVICE.compile_attempts(&dev0), 1);
        assert_eq!(PER_DEVICE.compile_attempts(&dev1), 1);
    }

    const RACED: KernelSource = KernelSource {
        name: "sc24137_seam_test_raced",
        src: "extern \"C\" __global__ void raced(float* out) { out[0] = 2.0f; }",
        cc_floor: (7, 0),
    };

    /// Callers racing on one key share one compile (the per-key slot, not the map lock, is what
    /// serialises them).
    #[test]
    fn racing_callers_share_one_compile() {
        let Some(dev) = device() else { return };
        let modules: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| RACED.compiled(&dev).expect("compiles")))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(modules.iter().all(|m| Arc::ptr_eq(m, &modules[0])));
        assert_eq!(RACED.compile_attempts(&dev), 1);
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
