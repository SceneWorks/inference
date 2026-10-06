pub(crate) fn phys_footprint() -> (u64, u64) {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u8) -> i32;
    }
    // `rusage_info_v4` is 296 bytes: `ri_phys_footprint` at byte 72 and
    // `ri_lifetime_max_phys_footprint` at byte 240 (see `<sys/resource.h>`).
    let mut buffer = [0u8; 512];
    // SAFETY: the buffer is larger than `rusage_info_v4` and the pid is our own.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, buffer.as_mut_ptr()) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    let read = |offset: usize| u64::from_ne_bytes(buffer[offset..offset + 8].try_into().unwrap());
    (read(72), read(240))
}
