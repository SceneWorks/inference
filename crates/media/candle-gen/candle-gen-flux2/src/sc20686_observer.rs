//! Optional SC-20686 campaign observer. `None` is the production default.
use std::cell::RefCell;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct CacheEvent {
    pub phase: &'static str,
    pub attention: &'static str,
    pub persistent_bytes: u64,
    pub transient_bytes: u64,
    pub elapsed_ms: u64,
    pub reused: u64,
}
pub trait CacheObserver {
    fn record(&mut self, event: CacheEvent);
}
thread_local! { static ACTIVE: RefCell<Option<Box<dyn CacheObserver>>> = RefCell::new(None); }
pub struct Scope {
    started: Instant,
}
pub fn install(observer: Box<dyn CacheObserver>) -> Scope {
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(observer));
    Scope {
        started: Instant::now(),
    }
}
pub fn observe(phase: &'static str, persistent_bytes: u64, transient_bytes: u64, reused: u64) {
    ACTIVE.with(|slot| {
        if let Some(observer) = slot.borrow_mut().as_mut() {
            observer.record(CacheEvent {
                phase,
                attention: "cross",
                persistent_bytes,
                transient_bytes,
                elapsed_ms: 0,
                reused,
            });
        }
    });
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
        let _ = self.started;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Sink(std::rc::Rc<std::cell::RefCell<Vec<CacheEvent>>>);
    impl CacheObserver for Sink {
        fn record(&mut self, e: CacheEvent) {
            self.0.borrow_mut().push(e);
        }
    }
    #[test]
    fn optional_observer_is_injectable() {
        let out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let _scope = install(Box::new(Sink(out.clone())));
        observe("generation", 256, 32, 1);
        assert_eq!(out.borrow()[0].attention, "cross");
    }
}
