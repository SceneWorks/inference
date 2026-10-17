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
pub use budget::{
    check_headroom, dcae_stage_bytes, gemma_layer_bytes, sana_block_bytes, PROCESS_RESERVE_BYTES,
};

#[cfg(not(target_os = "ios"))]
thread_local! {
    static CPU_FIXTURE_HEADROOM: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
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
        check_headroom(available_bytes(), scratch, PROCESS_RESERVE_BYTES)
            .map_err(|e| Error::Unsupported(format!("{label}: {e}; scratch {scratch} bytes + engineering reserve {PROCESS_RESERVE_BYTES} bytes (not a process bound)")))?;
    }
    Ok(())
}

/// Explicit C API synchronization is needed in addition to eval: completed graph values must not
/// leave earlier command-buffer resources live across the next headroom check.
fn synchronize() -> Result<()> {
    let stream = mlx_rs::Stream::new();
    let status = unsafe { mlx_sys::mlx_synchronize(stream.as_ptr()) };
    if status != 0 {
        return Err(Error::Msg(
            "iOS quantum stream synchronization failed".into(),
        ));
    }
    Ok(())
}

pub fn finish(arrays: &[&Array]) -> Result<()> {
    if enabled() {
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        synchronize()?;
        crate::residency::drain_allocator_cache();
    }
    Ok(())
}

pub fn run(label: &str, scratch: u64, build: impl FnOnce() -> Result<Array>) -> Result<Array> {
    if !enabled() {
        return build();
    }
    budget::guarded(
        available_bytes(),
        scratch,
        PROCESS_RESERVE_BYTES,
        build,
        |value| {
            mlx_rs::transforms::eval([value])?;
            synchronize()
        },
        || {
            crate::residency::drain_allocator_cache();
        },
    )
    .map_err(|error: Error| match error {
        Error::Msg(reason)
            if reason.starts_with("iOS quantum refused")
                || reason == "iOS quantum accounting overflow" =>
        {
            Error::Unsupported(format!("{label}: {reason}"))
        }
        other => other,
    })
}

/// Pre-load floor from the actual safetensors inventory, plus the largest tensor as read/copy
/// scratch. Called BEFORE model construction. Counts the entire VAE file even for txt2img.
/// Not a process upper bound: conversion or a backend operation can need more than this floor.
pub fn before_load(label: &str, components: &[std::path::PathBuf]) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    let mut total = 0u64;
    let mut largest = 0u64;
    for component in components {
        for header in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(component)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            total = total
                .checked_add(header.data_bytes)
                .ok_or_else(|| Error::Unsupported("iOS load accounting overflow".into()))?;
            largest = largest.max(header.data_bytes);
        }
    }
    let scratch = total
        .checked_add(largest)
        .ok_or_else(|| Error::Unsupported("iOS load accounting overflow".into()))?;
    before(label, scratch)
}

#[cfg(all(test, target_os = "macos"))]
#[path = "../mlx-gen-iris/tests/common/metal_guard.rs"]
mod quantum_metal_guard;

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
            sana_block_bytes(1, 16, 8, 5),
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
}
