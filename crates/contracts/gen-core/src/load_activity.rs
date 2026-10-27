//! Observational progress for synchronous work inside lazy provider loaders.
//!
//! A generator's public residency phases can contain several component loads. Scopes report
//! the operation actually executing without changing residency, source validation, or cache keys.
//! Install an observer on the thread that executes the provider, not on an async caller thread.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadActivity {
    /// Actual content hashing; metadata checks and memo hits must not enter this scope.
    VerifyingFiles,
    TextEncoder,
    Renderer,
    EncodingPrompt,
}

type Observer = Arc<dyn Fn(Option<LoadActivity>) + Send + Sync>;

#[derive(Default)]
struct State {
    observer: Option<Observer>,
    activity: Option<LoadActivity>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Request-local observation of provider work. The guard is thread-bound and restores any
/// enclosing observer on drop, including error and panic exits. Guards must be dropped in order.
#[must_use = "retain the observer guard for the duration of the request"]
pub struct LoadActivityObserver {
    previous: State,
    _thread_bound: PhantomData<Rc<()>>,
}

impl LoadActivityObserver {
    pub fn new(observe: impl Fn(Option<LoadActivity>) + Send + Sync + 'static) -> Self {
        let previous = STATE.with(|state| {
            state.replace(State {
                observer: Some(Arc::new(observe)),
                activity: None,
            })
        });
        Self {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for LoadActivityObserver {
    fn drop(&mut self) {
        STATE.with(|state| {
            state.replace(std::mem::take(&mut self.previous));
        });
    }
}

/// Report an operation for precisely its synchronous lifetime. Nested verification restores
/// the enclosing component's message when hashing finishes. No observer means no wire events.
#[must_use = "retain the activity guard until the operation finishes"]
pub struct LoadActivityScope {
    previous: Option<LoadActivity>,
    observer: Option<Observer>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl LoadActivityScope {
    pub fn enter(activity: LoadActivity) -> Self {
        let (previous, observer) = STATE.with(|state| {
            let mut state = state.borrow_mut();
            let previous = state.activity.replace(activity);
            (previous, state.observer.clone())
        });
        if let Some(observe) = &observer {
            observe(Some(activity));
        }
        Self {
            previous,
            observer,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for LoadActivityScope {
    fn drop(&mut self) {
        STATE.with(|state| state.borrow_mut().activity = self.previous);
        if let Some(observe) = &self.observer {
            observe(self.previous);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn nested_hashing_restores_the_component_and_isolates_threads() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let _observer =
            LoadActivityObserver::new(move |event| captured.lock().unwrap().push(event));
        {
            let _text = LoadActivityScope::enter(LoadActivity::TextEncoder);
            {
                let _hash = LoadActivityScope::enter(LoadActivity::VerifyingFiles);
            }
            std::thread::spawn(|| {
                let _other = LoadActivityScope::enter(LoadActivity::Renderer);
            })
            .join()
            .unwrap();
        }
        assert_eq!(
            *events.lock().unwrap(),
            [
                Some(LoadActivity::TextEncoder),
                Some(LoadActivity::VerifyingFiles),
                Some(LoadActivity::TextEncoder),
                None
            ]
        );
    }

    #[test]
    fn panic_unwinds_activity_and_does_not_leak_observers_to_the_next_request() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let _ = std::panic::catch_unwind(|| {
            let _observer =
                LoadActivityObserver::new(move |event| captured.lock().unwrap().push(event));
            let _hash = LoadActivityScope::enter(LoadActivity::VerifyingFiles);
            panic!("failed source read");
        });
        let _next = LoadActivityScope::enter(LoadActivity::Renderer);
        assert_eq!(
            *events.lock().unwrap(),
            [Some(LoadActivity::VerifyingFiles), None]
        );
    }

    #[test]
    fn actual_hash_reads_report_verification_but_memo_hits_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.bin");
        std::fs::write(&file, [42; 8192]).unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let _observer =
            LoadActivityObserver::new(move |event| captured.lock().unwrap().push(event));
        let first = crate::ArtifactSeal::pin(&file).unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            [Some(LoadActivity::VerifyingFiles), None]
        );
        events.lock().unwrap().clear();
        let warm = crate::ArtifactSeal::pin(&file).unwrap();
        assert_eq!(warm.content_sha256(), first.content_sha256());
        assert!(
            events.lock().unwrap().is_empty(),
            "a cached digest is not a hash read"
        );
    }
}
