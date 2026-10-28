//! Physical host availability for MOSS-owned encoded/decoder staging. This is separate from CUDA
//! VRAM and never substitutes host RAM for a device budget. No UI knob, cache mode or measurement.
use crate::{Error, Result};
fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("MOSS host memory discovery: {message}"))
}

/// Available physical host bytes. The platform probe and its one documented formula live in
/// `core_llm::available_host_memory_bytes` (sc-27317), so LLM admission and MOSS staging read
/// the same number; this keeps MOSS's typed error and non-zero requirement.
pub fn available_bytes() -> Result<u64> {
    crate::core_llm::available_host_memory_bytes()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid("available host bytes unavailable on this platform"))
}

/// Actual Apple unified-memory physical TOTAL, distinct from available pages and Metal's
/// recommended working set. Used only when the SDK requests a fraction; no allocator/device call.
#[cfg(target_os = "macos")]
pub fn total_bytes() -> Result<u64> {
    let mut total = 0_u64;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: the name is a NUL-terminated C-string literal; `total` and `size` are live locals,
    // `size` holding the 8-byte capacity of `total`, so the kernel writes at most that many bytes;
    // the new-value pointer is null with length 0 (read-only query).
    let status = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || size != std::mem::size_of::<u64>() || total == 0 {
        return Err(invalid("physical unified total unavailable"));
    }
    Ok(total)
}
#[cfg(not(target_os = "macos"))]
pub fn total_bytes() -> Result<u64> {
    Err(invalid("physical unified total is supported only on macOS"))
}
