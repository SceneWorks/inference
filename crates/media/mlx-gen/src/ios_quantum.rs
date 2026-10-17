//! Bounded SANA/Gemma execution quanta on a physical iOS device.
//!
//! Before constructing each graph, compare architecture-derived tensor scratch plus an explicit
//! engineering reserve with CURRENT OS headroom. Materialize and synchronize each result, then
//! drain reusable buffers before the next quantum. This limits lazy graph/cast overlap; it is NOT
//! an allocator interception or an absolute jetsam guarantee. Driver allocations, OS pressure and
//! a single backend operation can exceed the accounting. Production artifact admission is unchanged.

use crate::{Error, Result};
use mlx_rs::Array;
#[path = "ios_quantum_budget.rs"]
mod budget;
#[cfg(not(target_os = "ios"))]
pub use budget::measured_peak_bytes;
pub use budget::{
    allocated_bytes, allocated_tensor_bytes, assert_tracks_measured, attention_block_bytes,
    check_headroom, conv_stage_bytes, elementwise_bytes, rope_table_bytes, sampler_step_bytes,
    scores_bytes, tensor_bytes, AttentionBlockGeometry, ConvStageGeometry, Scores,
    PROCESS_RESERVE_BYTES,
};

/// A typed record of the quantum that refused BEFORE constructing its graph. Error values cross
/// provider/contract seams that may stringify them, so the refusal is ALSO recorded here, where a
/// device capture reads it without parsing any message. Capacity evidence, never a cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantumRefusal {
    pub label: String,
    pub scratch_bytes: u64,
    pub reserve_bytes: u64,
    /// Live `os_proc_available_memory` at the check; `None` when the OS reported none.
    pub available_bytes: Option<u64>,
}

static LAST_REFUSAL: std::sync::Mutex<Option<QuantumRefusal>> = std::sync::Mutex::new(None);

fn record_refusal(label: &str, scratch: u64, available: Option<u64>) {
    if let Ok(mut slot) = LAST_REFUSAL.lock() {
        *slot = Some(QuantumRefusal {
            label: label.to_string(),
            scratch_bytes: scratch,
            reserve_bytes: PROCESS_RESERVE_BYTES,
            available_bytes: available,
        });
    }
}

/// Take (and clear) the most recent pre-construction quantum refusal in this process.
pub fn take_refusal() -> Option<QuantumRefusal> {
    LAST_REFUSAL.lock().ok().and_then(|mut slot| slot.take())
}

/// One observed quantum boundary in a CPU diagnostic fixture.
#[cfg(not(target_os = "ios"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantumEvent {
    /// A pre-construction headroom check passed for `scratch` bytes.
    Checked { label: String, scratch: u64 },
    /// A completed quantum synchronized MLX stream index `stream`.
    Synchronized { label: String, stream: i32 },
}

#[cfg(not(target_os = "ios"))]
type QuantumObserver = Box<dyn FnMut(&QuantumEvent)>;

#[cfg(not(target_os = "ios"))]
thread_local! {
    static CPU_FIXTURE_HEADROOM: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    static CPU_FIXTURE_OBSERVER: std::cell::RefCell<Option<QuantumObserver>> = const { std::cell::RefCell::new(None) };
}

/// Observe checks/synchronizations of a CPU diagnostic fixture. Unavailable on iOS.
#[cfg(not(target_os = "ios"))]
pub fn with_cpu_diagnostic_observer<T>(
    observer: impl FnMut(&QuantumEvent) + 'static,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    struct Restore(Option<QuantumObserver>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CPU_FIXTURE_OBSERVER.with(|s| *s.borrow_mut() = previous);
        }
    }
    let previous = CPU_FIXTURE_OBSERVER.with(|s| s.borrow_mut().replace(Box::new(observer)));
    let _restore = Restore(previous);
    body()
}

fn notify(_event: impl FnOnce() -> QuantumEventArg) {
    #[cfg(not(target_os = "ios"))]
    {
        let Some(mut observer) = CPU_FIXTURE_OBSERVER.with(|s| s.borrow_mut().take()) else {
            return;
        };
        let event = match _event() {
            QuantumEventArg::Checked(label, scratch) => QuantumEvent::Checked { label, scratch },
            QuantumEventArg::Synchronized(label, stream) => {
                QuantumEvent::Synchronized { label, stream }
            }
        };
        observer(&event);
        CPU_FIXTURE_OBSERVER.with(|s| {
            let mut slot = s.borrow_mut();
            if slot.is_none() {
                *slot = Some(observer);
            }
        });
    }
}

// Fields are only read by the host CPU-fixture observer; on iOS `notify` discards them.
#[cfg_attr(target_os = "ios", allow(dead_code))]
enum QuantumEventArg {
    Checked(String, u64),
    Synchronized(String, i32),
}

/// Exercise the exact physical-iOS barriers on tiny CPU models, with deterministic headroom.
/// Unavailable on iOS: a diagnostic fixture cannot replace a phone's real OS reading.
#[cfg(not(target_os = "ios"))]
pub fn with_cpu_diagnostic_headroom<T>(bytes: u64, body: impl FnOnce() -> Result<T>) -> Result<T> {
    if mlx_rs::Device::try_default()? != mlx_rs::Device::cpu() {
        return Err(Error::Unsupported(
            "quantum fixture requires an explicit CPU device".into(),
        ));
    }
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CPU_FIXTURE_HEADROOM.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(CPU_FIXTURE_HEADROOM.with(|s| s.replace(Some(bytes))));
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), body)
}

pub fn enabled() -> bool {
    #[cfg(not(target_os = "ios"))]
    if CPU_FIXTURE_HEADROOM.with(|s| s.get().is_some()) {
        return true;
    }
    cfg!(all(target_os = "ios", not(target_abi = "sim")))
}

/// Live remaining capacity, never interpreted as a measured total cap.
pub fn available_bytes() -> Option<u64> {
    #[cfg(not(target_os = "ios"))]
    if let Some(bytes) = CPU_FIXTURE_HEADROOM.with(|s| s.get()) {
        return Some(bytes);
    }
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        unsafe extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        let bytes = unsafe { os_proc_available_memory() } as u64;
        (bytes > 0).then_some(bytes)
    }
    #[cfg(not(all(target_os = "ios", not(target_abi = "sim"))))]
    {
        None
    }
}

/// Cache lifetime matches the enclosing SANA/Gemma operation, including errors and early returns.
/// Nested scopes restore zero; the outer scope restores the caller's previous cache policy.
pub struct Scope(Option<usize>);
impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}
impl Scope {
    pub fn new() -> Self {
        if enabled() {
            let previous = mlx_rs::memory::set_cache_limit(0);
            crate::residency::drain_allocator_cache();
            Self(Some(previous))
        } else {
            Self(None)
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            crate::residency::drain_allocator_cache();
            mlx_rs::memory::set_cache_limit(previous);
        }
    }
}

pub fn before(label: &str, scratch: u64) -> Result<()> {
    if enabled() {
        let available = available_bytes();
        check_headroom(available, scratch, PROCESS_RESERVE_BYTES).map_err(|e| {
            record_refusal(label, scratch, available);
            Error::Unsupported(format!("{label}: {e}; scratch {scratch} bytes + engineering reserve {PROCESS_RESERVE_BYTES} bytes (not a process bound)"))
        })?;
        notify(|| QuantumEventArg::Checked(label.to_string(), scratch));
    }
    Ok(())
}

/// Explicit C API synchronization is needed in addition to eval: completed graph values must not
/// leave earlier command-buffer resources live across the next headroom check. Synchronize the
/// stream this quantum executed on: a caller-installed task-local stream, else the default.
fn synchronize(label: &str) -> Result<()> {
    let stream = mlx_rs::Stream::task_local_or_default();
    let status = unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()) };
    if status != 0 {
        return Err(Error::Msg(
            "iOS quantum stream synchronization failed".into(),
        ));
    }
    let index = stream.get_index().unwrap_or(-1);
    notify(|| QuantumEventArg::Synchronized(label.to_string(), index));
    Ok(())
}

pub fn finish(label: &str, arrays: &[&Array]) -> Result<()> {
    if enabled() {
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        synchronize(label)?;
        crate::residency::drain_allocator_cache();
    }
    Ok(())
}

pub fn run(label: &str, scratch: u64, build: impl FnOnce() -> Result<Array>) -> Result<Array> {
    if !enabled() {
        return build();
    }
    let checked = std::cell::Cell::new(false);
    let available = available_bytes();
    let result = budget::guarded(
        available,
        scratch,
        PROCESS_RESERVE_BYTES,
        || {
            checked.set(true);
            notify(|| QuantumEventArg::Checked(label.to_string(), scratch));
            build()
        },
        |value| {
            mlx_rs::transforms::eval([value])?;
            synchronize(label)
        },
        || {
            crate::residency::drain_allocator_cache();
        },
    )
    .map_err(|error: Error| match error {
        Error::Msg(reason)
            if !checked.get()
                && (reason.starts_with("iOS quantum refused")
                    || reason == "iOS quantum accounting overflow") =>
        {
            record_refusal(label, scratch, available);
            Error::Unsupported(format!("{label}: {reason}"))
        }
        other => other,
    });
    result
}

/// Pre-load floor from the actual safetensors inventory, plus the largest tensor as read/copy
/// scratch. Called BEFORE model construction. Counts the entire VAE file even for txt2img.
/// Not a process upper bound: conversion or a backend operation can need more than this floor.
pub fn before_load(label: &str, components: &[std::path::PathBuf]) -> Result<()> {
    before_load_where(label, components, |_| true)
}

/// [`before_load`] over only the tensors `resident(name)` keeps: for a load that constructs just
/// part of a file up front (e.g. a rung-4 streamed DiT, whose block bodies are read per window by
/// that window's own priced quantum). Header-only; a no-op off device.
pub fn before_load_where(
    label: &str,
    components: &[std::path::PathBuf],
    resident: impl Fn(&str) -> bool,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    before(label, load_floor_bytes(components, resident)?)
}

/// Header-only load floor: the stored bytes of every kept tensor. Proven against MLX's measured
/// peak while a file is read (`cpu_quantum_materializes_before_return_refuses_before_build_and_
/// restores_cache`): the bytes land once, with no extra per-tensor read copy.
pub fn load_floor_bytes(
    components: &[std::path::PathBuf],
    resident: impl Fn(&str) -> bool,
) -> Result<u64> {
    let mut total = 0u64;
    for component in components {
        for header in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(component)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            if resident(&header.name) {
                total = total
                    .checked_add(header.data_bytes)
                    .ok_or_else(|| Error::Unsupported("iOS load accounting overflow".into()))?;
            }
        }
    }
    Ok(total)
}

/// [`before_load`] for a component whose loader CASTS every tensor to a `bytes_per_element`-wide
/// dtype after reading it (e.g. an upcast to f32): the stored bytes AND the cast copies are live
/// together (proven against the measured peak of a read + cast). Header-only; a no-op off device.
pub fn before_load_cast(
    label: &str,
    components: &[std::path::PathBuf],
    bytes_per_element: u64,
) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    let overflow = || Error::Unsupported("iOS load accounting overflow".into());
    let mut total = 0u64;
    for component in components {
        for header in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(component)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            let elements = header
                .shape
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d as u64))
                .ok_or_else(overflow)?;
            let cast = elements
                .checked_mul(bytes_per_element)
                .ok_or_else(overflow)?;
            total = total
                .checked_add(header.data_bytes)
                .and_then(|t| t.checked_add(cast))
                .ok_or_else(overflow)?;
        }
    }
    before(label, total)
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../mlx-gen-iris/tests/common/metal_guard.rs"]
pub(crate) mod quantum_metal_guard;

#[cfg(all(test, target_os = "macos"))]
#[test]
fn cpu_quantum_materializes_before_return_refuses_before_build_and_restores_cache() {
    const CHILD: &str = "MLX_IOS_QUANTUM_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "ios_quantum::cpu_quantum_materializes_before_return_refuses_before_build_and_restores_cache", "--nocapture"])
            .env(CHILD,"1").status().unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    assert!(quantum_metal_guard::metal_forbidden());
    fn available(value: &Array) -> bool {
        let mut result = false;
        assert_eq!(
            unsafe { mlx_sys::_mlx_array_is_available(&mut result, value.as_ptr()) },
            0
        );
        result
    }
    let a = Array::from_slice(&[1.0f32; 16], &[4, 4]);
    let lazy = mlx_rs::ops::add(&a, &a).unwrap();
    assert!(
        !available(&lazy),
        "fixture must contain a genuinely lazy operation"
    );
    let old_cache = mlx_rs::memory::set_cache_limit(1234567);
    with_cpu_diagnostic_headroom(1024 * 1024 * 1024, || {
        let _scope = Scope::new();
        assert_eq!(mlx_rs::memory::set_cache_limit(0), 0);
        let out = run("tiny lazy graph", 4096, || Ok(lazy))?;
        assert!(
            available(&out),
            "quantum returned its lazy graph rather than materializing it"
        );
        assert_eq!(out.as_slice::<f32>(), &[2.0f32; 16]);
        // Real mixed-dtype convolution: F16 storage must not hide an F32 promoted copy.
        let input = Array::ones::<f32>(&[1, 4, 4, 8])?;
        let weight = Array::ones::<f32>(&[40, 1, 1, 8])?.as_dtype(mlx_rs::Dtype::Float16)?;
        let promoted = run(
            "tiny F16 convolution",
            // Promoted F32 weights (40*8 elements) plus the f32 output, counted for this op.
            conv_stage_bytes(ConvStageGeometry {
                batch: 1,
                height: 4,
                width: 4,
                in_channels: 8,
                out_channels: 40,
                element_bytes: 4,
                in_live: 1,
                out_live: 1,
                weight_bytes: 40 * 8 * 4,
            }),
            || {
                Ok(mlx_rs::ops::conv2d(
                    &input,
                    &weight,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                    1,
                )?)
            },
        )?;
        assert_eq!(promoted.dtype(), mlx_rs::Dtype::Float32);
        assert!(available(&promoted));
        Ok(())
    })
    .unwrap();
    assert_eq!(mlx_rs::memory::set_cache_limit(old_cache), 1234567);
    let constructed = std::cell::Cell::new(false);
    let refused = with_cpu_diagnostic_headroom(0, || {
        run("refused", 4096, || {
            constructed.set(true);
            Ok(a.clone())
        })
    });
    assert!(refused.is_err());
    assert!(!constructed.get());
    // The refusal is typed for a capture, independent of how the error string is propagated.
    let record = take_refusal().expect("a pre-construction refusal is recorded");
    assert_eq!(record.label, "refused");
    assert_eq!(record.scratch_bytes, 4096);
    assert_eq!(record.available_bytes, Some(0));
    assert_eq!(take_refusal(), None, "taking clears the record");
    let before_refused = with_cpu_diagnostic_headroom(1, || before("before refused", 8));
    assert!(before_refused.is_err());
    assert_eq!(
        take_refusal().map(|r| (r.label, r.available_bytes)),
        Some(("before refused".to_string(), Some(1)))
    );
    // The measurement every counted formula is proven against: two 256x256 f32 inputs built
    // inside the quantum plus their sum = three live 256 KiB tensors at the peak.
    let measured = measured_peak_bytes(|| {
        let a = Array::ones::<f32>(&[256, 256])?;
        let b = Array::ones::<f32>(&[256, 256])?;
        Ok(vec![mlx_rs::ops::add(&a, &b)?])
    })
    .unwrap();
    let three = 3 * tensor_bytes(4, &[256, 256]);
    assert_tracks_measured("ones + ones", three, measured, 1.5);
    // The load floor is proven against MLX's peak while a safetensors file is read: the stored
    // bytes land once, with no extra per-tensor copy (no "largest tensor" scratch on top).
    {
        let dir = std::env::temp_dir().join(format!("ios-quantum-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("w.safetensors");
        let a = Array::ones::<f32>(&[256, 1024]).unwrap();
        let b = Array::ones::<f32>(&[512, 1024]).unwrap();
        let c = Array::ones::<f32>(&[1024, 1024]).unwrap();
        Array::save_safetensors(vec![("a", &a), ("b", &b), ("c", &c)], None, &file).unwrap();
        let measured = measured_peak_bytes(|| {
            let w = crate::weights::Weights::from_file(&file)?;
            Ok(["a", "b", "c"]
                .iter()
                .map(|k| w.require(k).unwrap().clone())
                .collect())
        })
        .unwrap();
        // Read + upcast: stored f16 bytes and the f32 copies are live together.
        let half = dir.join("h.safetensors");
        let h = [&a, &b, &c].map(|x| x.as_dtype(mlx_rs::Dtype::Float16).unwrap());
        Array::save_safetensors(vec![("a", &h[0]), ("b", &h[1]), ("c", &h[2])], None, &half)
            .unwrap();
        let cast_measured = measured_peak_bytes(|| {
            let w = crate::weights::Weights::from_file(&half)?;
            Ok(["a", "b", "c"]
                .iter()
                .map(|k| {
                    w.require(k)
                        .unwrap()
                        .as_dtype(mlx_rs::Dtype::Float32)
                        .unwrap()
                })
                .collect())
        })
        .unwrap();
        let stored_half = (256 + 512 + 1024) * 1024 * 2;
        assert_tracks_measured(
            "safetensors load + f32 cast",
            stored_half + 2 * stored_half,
            cast_measured,
            1.5,
        );
        assert_tracks_measured(
            "safetensors load",
            load_floor_bytes(std::slice::from_ref(&file), |_| true).unwrap(),
            measured,
            1.5,
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
    // A casting load prices stored bytes + the widened copies + the largest widened tensor.
    let dir = std::env::temp_dir().join(format!("ios-quantum-cast-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("vae.safetensors");
    let header = r#"{"w":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.resize(bytes.len() + 64, 0);
    std::fs::write(&file, bytes).unwrap();
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = seen.clone();
    with_cpu_diagnostic_headroom(1 << 30, || {
        with_cpu_diagnostic_observer(
            move |e| sink.borrow_mut().push(e.clone()),
            || {
                before_load("stored", std::slice::from_ref(&file))?;
                before_load_cast("cast", std::slice::from_ref(&file), 4)
            },
        )
    })
    .unwrap();
    assert_eq!(
        *seen.borrow(),
        [
            QuantumEvent::Checked {
                label: "stored".into(),
                scratch: 64
            },
            QuantumEvent::Checked {
                label: "cast".into(),
                scratch: 64 + 128
            },
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(all(test, target_os = "macos"))]
#[test]
fn cpu_quantum_synchronizes_the_installed_non_default_stream() {
    const CHILD: &str = "MLX_IOS_QUANTUM_STREAM_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ios_quantum::cpu_quantum_synchronizes_the_installed_non_default_stream",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(quantum_metal_guard::forbid_metal());
    let installed = mlx_rs::Stream::new_with_device(&mlx_rs::Device::cpu());
    let default_index = mlx_rs::Stream::new().get_index().unwrap();
    let installed_index = installed.get_index().unwrap();
    assert_ne!(
        installed_index, default_index,
        "fixture needs a genuinely non-default stream"
    );
    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = seen.clone();
    with_cpu_diagnostic_headroom(1024 * 1024 * 1024, || {
        with_cpu_diagnostic_observer(
            move |event| sink.borrow_mut().push(event.clone()),
            || {
                mlx_rs::with_new_default_stream(installed.clone(), || {
                    let a = Array::from_slice(&[1.0f32; 4], &[4]);
                    run("non-default stream", 4096, || Ok(mlx_rs::ops::add(&a, &a)?))?;
                    finish("non-default finish", &[&a])
                })
            },
        )
    })
    .unwrap();
    let synchronized: Vec<i32> = seen
        .borrow()
        .iter()
        .filter_map(|event| match event {
            QuantumEvent::Synchronized { stream, .. } => Some(*stream),
            QuantumEvent::Checked { .. } => None,
        })
        .collect();
    assert_eq!(synchronized, [installed_index, installed_index]);
}
