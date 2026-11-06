//! One bounded reclaim attempt; the allocator's cap and high-water marks never change.

use crate::Result;

pub(super) struct State {
    pub cap_bytes: u64,
    pub used_bytes: u64,
}

pub(super) trait Reclaim {
    fn snapshot(&mut self, phase: &str) -> Result<State>;
    fn trim(&mut self) -> Result<()>;
}

pub(super) fn reclaim_and_retry<T, E>(
    requested_bytes: u64,
    cap_bytes: u64,
    reclaim: &mut impl Reclaim,
    retry: impl FnOnce() -> std::result::Result<T, E>,
) -> Result<Option<std::result::Result<T, E>>> {
    if requested_bytes == 0 || requested_bytes > cap_bytes {
        return Ok(None);
    }
    // Never synchronize here: another stream in this context may be capturing.
    // Pending frees may still count as outstanding; TrimTo releases only unused backing.
    let live = reclaim.snapshot("observed")?;
    if live.cap_bytes != cap_bytes {
        crate::bail!("CUDA proof pool changed during OOM recovery");
    }
    if live
        .used_bytes
        .checked_add(requested_bytes)
        .is_none_or(|needed| needed > cap_bytes)
    {
        return Ok(None);
    }
    reclaim.trim()?;
    let trimmed = reclaim.snapshot("trimmed")?;
    if trimmed.cap_bytes != cap_bytes {
        crate::bail!("CUDA proof pool changed during OOM trim");
    }
    if trimmed
        .used_bytes
        .checked_add(requested_bytes)
        .is_none_or(|needed| needed > cap_bytes)
    {
        return Ok(None);
    }
    // The closure invokes the same original allocation once. No alternate allocator,
    // unbounded loop, pool replacement or watermark reset is a recovery mechanism.
    Ok(Some(retry()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct Fake {
        trace: Rc<RefCell<Vec<String>>>,
        used: u64,
        cap: u64,
        fail: Option<&'static str>,
        trimmed_used: Option<u64>,
    }
    impl Fake {
        fn event(&self, event: &str) -> Result<()> {
            self.trace.borrow_mut().push(event.into());
            if self.fail == Some(event) {
                crate::bail!("fake {event} failure");
            }
            Ok(())
        }
    }
    impl Reclaim for Fake {
        fn snapshot(&mut self, phase: &str) -> Result<State> {
            self.event(phase)?;
            Ok(State {
                cap_bytes: self.cap,
                used_bytes: if phase == "trimmed" {
                    self.trimmed_used.unwrap_or(self.used)
                } else {
                    self.used
                },
            })
        }
        fn trim(&mut self) -> Result<()> {
            self.event("trim")
        }
    }
    fn fake(used: u64) -> Fake {
        Fake {
            trace: Rc::default(),
            used,
            cap: 512,
            fail: None,
            trimmed_used: None,
        }
    }

    #[test]
    fn allocation_recovery_observes_then_trims_then_retries_once_without_changing_cap() {
        let mut fake = fake(256);
        let trace = fake.trace.clone();
        assert_eq!(
            reclaim_and_retry(256, 512, &mut fake, || {
                trace.borrow_mut().push("retry".into());
                Ok::<_, &str>(42)
            })
            .unwrap(),
            Some(Ok(42))
        );
        assert_eq!(*trace.borrow(), ["observed", "trim", "trimmed", "retry"]);
        assert_eq!(fake.cap, 512);
    }

    #[test]
    fn allocation_recovery_does_not_attempt_to_fit_live_or_oversize_allocations() {
        for (used, requested) in [(512, 1), (257, 256), (0, 513), (0, 0), (u64::MAX, 1)] {
            let mut fake = fake(used);
            let result = reclaim_and_retry(
                requested,
                512,
                &mut fake,
                || -> std::result::Result<(), ()> { panic!("unfitting request retried") },
            )
            .unwrap();
            assert!(result.is_none());
            assert!(!fake.trace.borrow().iter().any(|event| event == "trim"));
        }
    }

    #[test]
    fn allocation_recovery_propagates_observation_and_trim_failures_without_retry() {
        for phase in ["observed", "trim", "trimmed"] {
            let mut fake = fake(256);
            fake.fail = Some(phase);
            assert!(
                reclaim_and_retry(256, 512, &mut fake, || -> std::result::Result<(), ()> {
                    panic!("failed reclaim retried")
                })
                .is_err()
            );
            assert_eq!(fake.trace.borrow().last().map(String::as_str), Some(phase));
        }
    }

    #[test]
    fn allocation_recovery_refuses_changed_pool_and_does_not_loop_on_retry_failure() {
        let mut changed = fake(0);
        changed.cap = 1024;
        assert!(
            reclaim_and_retry(256, 512, &mut changed, || -> std::result::Result<(), ()> {
                panic!("changed pool retried")
            })
            .is_err()
        );
        let mut fake = fake(256);
        let trace = fake.trace.clone();
        assert_eq!(
            reclaim_and_retry(256, 512, &mut fake, || {
                trace.borrow_mut().push("retry".into());
                Err::<(), _>("still out of memory")
            })
            .unwrap(),
            Some(Err("still out of memory"))
        );
        assert_eq!(
            trace
                .borrow()
                .iter()
                .filter(|event| *event == "retry")
                .count(),
            1
        );
    }

    #[test]
    fn allocation_recovery_rechecks_live_bytes_after_trim_before_retry() {
        let mut fake = fake(256);
        fake.trimmed_used = Some(512);
        assert!(
            reclaim_and_retry(256, 512, &mut fake, || -> std::result::Result<(), ()> {
                panic!("late allocation pressure retried")
            })
            .unwrap()
            .is_none()
        );
    }
}
