//! Physical host availability for MOSS-owned encoded/decoder staging. This is separate from CUDA
//! VRAM and never substitutes host RAM for a device budget. No UI knob, cache mode or measurement.
use crate::{Error, Result};
fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("MOSS host memory discovery: {message}"))
}

#[cfg(target_os = "linux")]
pub fn available_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let kib = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .ok_or_else(|| invalid("MemAvailable missing"))?
        .split_whitespace()
        .next()
        .ok_or_else(|| invalid("MemAvailable empty"))?
        .parse::<u64>()
        .map_err(invalid)?;
    kib.checked_mul(1024)
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("available host bytes overflow/zero"))
}
#[cfg(target_os = "windows")]
pub fn available_bytes() -> Result<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.ullAvailPhys == 0 {
        return Err(invalid(std::io::Error::last_os_error()));
    }
    Ok(status.ullAvailPhys)
}
#[cfg(target_os = "macos")]
#[allow(deprecated)]
pub fn available_bytes() -> Result<u64> {
    let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let page_size = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .map_err(|_| invalid("host page size unavailable"))?;
    let host = unsafe { libc::mach_host_self() };
    let result = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut info as *mut _ as libc::host_info64_t,
            &mut count,
        )
    };
    unsafe { libc::mach_port_deallocate(libc::mach_task_self(), host) };
    if result != libc::KERN_SUCCESS || page_size == 0 || count != libc::HOST_VM_INFO64_COUNT {
        return Err(invalid("host VM page metadata unavailable"));
    }
    // Kernel free_count includes speculative. Credit reclaimable file pages, not anonymous
    // inactive pages; speculative read-ahead is already included in the kernel free list.
    let reclaimable = info
        .external_page_count
        .saturating_sub(info.speculative_count);
    u64::from(info.free_count)
        .checked_add(u64::from(info.purgeable_count))
        .and_then(|n| n.checked_add(u64::from(reclaimable)))
        .and_then(|n| n.checked_mul(page_size))
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("available host bytes overflow/zero"))
}
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub fn available_bytes() -> Result<u64> {
    Err(invalid(
        "platform does not expose supported host availability",
    ))
}
