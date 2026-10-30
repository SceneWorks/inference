//! Request-scoped observations of actual prompt and generation work.
//!
//! Providers install the request observer on their execution thread. Shared tokenization and
//! decode seams report work only when it starts; reused conditioning does not synthesize phases.
//! Callbacks must enqueue/coalesce updates, never block on network or model work.

use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

/// Backend-neutral work currently executing for a text request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationPhase {
    /// Converting prompt text to token IDs (not loading model weights).
    EncodingPrompt,
    /// Encoding conditioning or feeding the uncached prompt into the model.
    Prefilling,
    /// Sampling or decoding the continuation after prompt work completes.
    Generating,
}

/// A request's optional progress sink, retained by clones of the request.
#[derive(Clone)]
pub struct GenerationProgressObserver(Arc<dyn Fn(Option<GenerationPhase>) + Send + Sync>);

impl std::fmt::Debug for GenerationProgressObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationProgressObserver")
            .finish_non_exhaustive()
    }
}

impl GenerationProgressObserver {
    /// Create an observer. `None` clears compute activity on completion or failure.
    pub fn new(callback: impl Fn(Option<GenerationPhase>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
}

#[derive(Clone)]
struct Active {
    observer: Option<GenerationProgressObserver>,
    phase: Option<GenerationPhase>,
}

thread_local! {
    static CURRENT: RefCell<Active> = const { RefCell::new(Active { observer: None, phase: None }) };
}

/// Thread-bound installation guard; dropping it clears this request and restores its caller.
///
/// Install even an absent observer to isolate nested requests from their caller's observer.
pub struct GenerationProgressGuard {
    previous: Active,
    _thread_bound: PhantomData<Rc<()>>,
}

impl crate::TextLlmRequest {
    /// Install this request's progress sink for shared work on the current execution thread.
    /// A provider must retain the guard through its entire generation, including errors.
    pub fn install_progress(&self) -> GenerationProgressGuard {
        let previous = CURRENT.with(|current| {
            current.replace(Active {
                observer: self.progress.clone(),
                phase: None,
            })
        });
        GenerationProgressGuard {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for GenerationProgressGuard {
    fn drop(&mut self) {
        let previous = self.previous.clone();
        let finished = CURRENT.with(|current| current.replace(previous.clone()));
        if let Some(observer) = finished.observer {
            (observer.0)(None);
        }
        if let Some(observer) = previous.observer {
            (observer.0)(previous.phase);
        }
    }
}

/// Report a phase at the actual shared work seam. Consecutive identical phases coalesce.
/// No observer means no callback; this never guesses progress from elapsed time.
pub fn report_generation_phase(phase: GenerationPhase) {
    set_phase(Some(phase));
}

fn set_phase(phase: Option<GenerationPhase>) {
    let observer = CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.phase == phase {
            return None;
        }
        current.phase = phase;
        current.observer.clone()
    });
    if let Some(observer) = observer {
        (observer.0)(phase);
    }
}

thread_local! {
    static BATCH: RefCell<Vec<Active>> = const { RefCell::new(Vec::new()) };
}

/// Request-indexed observations for shared batched compute. Indices refer to the submitted
/// batch, so providers must supply observers in the same order as their admitted decode requests.
/// This guard clears every admitted request on success, cancellation, or error.
pub struct GenerationBatchProgressGuard {
    previous: Vec<Active>,
    _thread_bound: PhantomData<Rc<()>>,
}

/// Install the admitted requests' sinks while the shared batch engine runs.
pub fn install_generation_batch_progress(
    observers: impl IntoIterator<Item = Option<GenerationProgressObserver>>,
) -> GenerationBatchProgressGuard {
    let active = observers
        .into_iter()
        .map(|observer| Active {
            observer,
            phase: None,
        })
        .collect();
    let previous = BATCH.with(|batch| batch.replace(active));
    GenerationBatchProgressGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

/// Report actual work on one admitted lane; waiting/cancelled lanes receive no fabricated prefill.
/// `None` clears a lane as soon as it retires, independently of the other batch members.
pub fn report_generation_phase_for(index: usize, phase: Option<GenerationPhase>) {
    let observer = BATCH.with(|batch| {
        let mut batch = batch.borrow_mut();
        let active = batch.get_mut(index)?;
        if active.phase == phase {
            return None;
        }
        active.phase = phase;
        active.observer.clone()
    });
    if let Some(observer) = observer {
        (observer.0)(phase);
    }
}

impl Drop for GenerationBatchProgressGuard {
    fn drop(&mut self) {
        let finished = BATCH.with(|batch| batch.replace(std::mem::take(&mut self.previous)));
        for active in finished {
            if let Some(observer) = active.observer {
                (observer.0)(None);
            }
        }
    }
}

/// A synchronous shared operation, restoring the surrounding phase even on failure/unwind.
/// Do not use this around construction of a lazy compute graph: report at dispatch and advance
/// only after its existing evaluation/synchronization boundary instead.
pub struct GenerationProgressScope {
    previous: Option<GenerationPhase>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl GenerationProgressScope {
    /// Enter actual synchronous work on the current request.
    pub fn enter(phase: GenerationPhase) -> Self {
        let previous = CURRENT.with(|current| current.borrow().phase);
        set_phase(Some(phase));
        Self {
            previous,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for GenerationProgressScope {
    fn drop(&mut self) {
        set_phase(self.previous);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn request(events: &Arc<Mutex<Vec<Option<GenerationPhase>>>>) -> crate::TextLlmRequest {
        let events = events.clone();
        crate::TextLlmRequest {
            progress: Some(GenerationProgressObserver::new(move |phase| {
                events.lock().unwrap().push(phase)
            })),
            ..Default::default()
        }
    }

    #[test]
    fn scopes_restore_phases_and_clear_errors_without_inventing_prefill() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let req = request(&events);
        let fail = || -> crate::Result<()> {
            let _request = req.install_progress();
            {
                let _encoding = GenerationProgressScope::enter(GenerationPhase::EncodingPrompt);
            }
            report_generation_phase(GenerationPhase::Generating); // reused/precomputed conditioning
            report_generation_phase(GenerationPhase::Generating); // coalesced
            Err(crate::Error::Canceled)
        };
        assert!(fail().is_err());
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                Some(GenerationPhase::EncodingPrompt),
                None,
                Some(GenerationPhase::Generating),
                None
            ]
        );
    }

    #[test]
    fn nested_unobserved_requests_do_not_borrow_the_callers_sink() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let req = request(&events);
        let _request = req.install_progress();
        report_generation_phase(GenerationPhase::Prefilling);
        {
            let _nested = crate::TextLlmRequest::default().install_progress();
            report_generation_phase(GenerationPhase::EncodingPrompt);
        }
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                Some(GenerationPhase::Prefilling),
                Some(GenerationPhase::Prefilling)
            ]
        );
    }

    #[test]
    fn batched_lanes_report_only_when_their_work_starts_and_clear_on_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let first = request(&events);
        let waiting = Arc::new(Mutex::new(Vec::new()));
        let second = request(&waiting);
        {
            let _batch = install_generation_batch_progress([first.progress, second.progress]);
            report_generation_phase_for(0, Some(GenerationPhase::Prefilling));
            report_generation_phase_for(0, Some(GenerationPhase::Generating));
            report_generation_phase_for(0, Some(GenerationPhase::Generating));
            assert!(
                waiting.lock().unwrap().is_empty(),
                "queued/cancelled lane has no prefill"
            );
        } // same unwind-safe cleanup runs on error
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                Some(GenerationPhase::Prefilling),
                Some(GenerationPhase::Generating),
                None
            ]
        );
        assert_eq!(*waiting.lock().unwrap(), vec![None]);
    }

    #[test]
    fn observer_isolation_survives_concurrent_requests_and_unwind() {
        let first = Arc::new(Mutex::new(Vec::new()));
        let first_req = request(&first);
        let second = Arc::new(Mutex::new(Vec::new()));
        let second_req = request(&second);
        let ready = Arc::new(std::sync::Barrier::new(2));
        let child_ready = ready.clone();
        let child = std::thread::spawn(move || {
            let _request = first_req.install_progress();
            let _ = std::panic::catch_unwind(|| {
                let _phase = GenerationProgressScope::enter(GenerationPhase::EncodingPrompt);
                child_ready.wait();
                child_ready.wait();
                panic!("fixture failure");
            });
        });
        {
            let _request = second_req.install_progress();
            report_generation_phase(GenerationPhase::Prefilling);
            ready.wait(); // both requests are active on different execution threads
            assert_eq!(
                *first.lock().unwrap(),
                vec![Some(GenerationPhase::EncodingPrompt)]
            );
            assert_eq!(
                *second.lock().unwrap(),
                vec![Some(GenerationPhase::Prefilling)]
            );
            report_generation_phase(GenerationPhase::Generating);
            ready.wait();
        }
        child.join().unwrap();
        report_generation_phase(GenerationPhase::Prefilling); // no request installed here
        assert_eq!(
            *first.lock().unwrap(),
            vec![Some(GenerationPhase::EncodingPrompt), None, None]
        );
        assert_eq!(
            *second.lock().unwrap(),
            vec![
                Some(GenerationPhase::Prefilling),
                Some(GenerationPhase::Generating),
                None
            ]
        );
    }
}
