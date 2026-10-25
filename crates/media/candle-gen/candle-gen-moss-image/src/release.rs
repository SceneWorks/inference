//! CUDA tensor drops enqueue frees; synchronize after those drops before observing free VRAM.

pub(crate) fn release_then_sync<T, E>(
    owned: T,
    synchronize: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    drop(owned);
    synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct QueuedFree<'a> {
        pending: &'a Cell<u64>,
        bytes: u64,
    }
    impl Drop for QueuedFree<'_> {
        fn drop(&mut self) {
            self.pending.set(self.pending.get() + self.bytes);
        }
    }

    #[test]
    fn released_components_reach_driver_before_next_admission() {
        let pending = Cell::new(0);
        let free = Cell::new(71);
        let weight = |bytes| QueuedFree {
            pending: &pending,
            bytes,
        };
        release_then_sync(
            (Some(weight(16)), Some(weight(10)), Some(weight(1))),
            || {
                // Model the stream-ordered allocator: dropping queues frees, and only a
                // later synchronization returns them to the driver-visible budget.
                free.set(free.get() + pending.replace(0));
                Ok::<_, ()>(())
            },
        )
        .unwrap();
        assert_eq!(pending.get(), 0);
        assert_eq!(free.get(), 98);
        assert!(
            free.get() >= 87,
            "unchanged next-request demand must fit after release"
        );
    }

    #[test]
    fn synchronization_failure_does_not_skip_resource_release() {
        let pending = Cell::new(0);
        let result = release_then_sync(
            QueuedFree {
                pending: &pending,
                bytes: 27,
            },
            || {
                assert_eq!(
                    pending.get(),
                    27,
                    "all frees precede even a failing synchronization"
                );
                Err("synchronization failed")
            },
        );
        assert_eq!(result, Err("synchronization failed"));
        assert_eq!(pending.get(), 27);
    }

    #[test]
    fn empty_state_still_drains_frees_from_request_local_tensors() {
        let pending = Cell::new(3);
        release_then_sync(None::<QueuedFree<'_>>, || {
            assert_eq!(pending.replace(0), 3);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(pending.get(), 0);
    }
}
