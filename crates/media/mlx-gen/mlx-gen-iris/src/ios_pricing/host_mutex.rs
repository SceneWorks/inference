//! Requested Rust allocation for one cold Apple pthread mutex initialization.
//!
//! Authority: libc 0.2.183 archive SHA-256
//! b5b646652bf6661599e1da8901b3b9522896f01e736bad5f723fe7a3a27f899d,
//! src/new/apple/libpthread/sys/_pthread/_pthread_types.rs:11-17,51-54;
//! src/primitives.rs:56-59 and src/macros.rs:156-158. The Apple64 C layout is
//! one i64 followed by 56 bytes, alignment 8. This is not sizeof(std::Mutex).
//!
//! Rust ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96 selects pthread on Apple;
//! std/src/sys/sync/mutex/pthread.rs::get boxes pal::Mutex, whose sole field is
//! UnsafeCell<libc::pthread_mutex_t>. The existing requested-host-layout premise
//! covers the repr(Rust) wrapper. This term excludes pthread internals, malloc
//! overhead, physical charge and every other metadata allocation.
//!
//! Count every lock attempt, including warm attempts, rather than crediting
//! OnceBox reuse or assuming that a concurrent initialization cannot lose a race.

pub(super) fn requested_bytes(lock_attempts: u64) -> Option<u64> {
    if !cfg!(all(target_vendor = "apple", target_pointer_width = "64"))
        || option_env!("IRIS_HOST_LAYOUT_COMPILER")
            != Some("ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96")
    {
        return None;
    }
    lock_attempts.checked_mul(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_mutex_requests_are_counted_without_reuse_credit() {
        if cfg!(all(target_vendor = "apple", target_pointer_width = "64"))
            && option_env!("IRIS_HOST_LAYOUT_COMPILER")
                == Some("ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96")
        {
            assert_eq!(requested_bytes(0), Some(0));
            assert_eq!(requested_bytes(1), Some(64));
            assert_eq!(requested_bytes(3), Some(192));
            assert_eq!(requested_bytes(u64::MAX), None);
        } else {
            assert_eq!(requested_bytes(1), None);
        }
    }
}
