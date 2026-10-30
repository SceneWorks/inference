//! The provider trait — the heart of the contract.

use crate::capabilities::TextLlmDescriptor;
use crate::error::Result;
use crate::output::{StreamEvent, TextLlmOutput};
use crate::request::TextLlmRequest;

/// Bounded execution-ordering state for a provider with a load stream and one
/// current execution stream. This owns at most TWO handles, not model tensors.
/// Fencing is NOT eviction or proof that an allocation/cache was released.
///
/// Backend handles stay opaque/tensor-neutral here. A failed stream switch keeps
/// the old handle and prevents new work; an unchanged stream adds no barrier.
pub struct OwnedExecutionFence<S: PartialEq> {
    load: S,
    execution: std::cell::RefCell<S>,
}

impl<S: PartialEq> OwnedExecutionFence<S> {
    pub fn new(load: S, execution: S) -> Self {
        Self {
            load,
            execution: std::cell::RefCell::new(execution),
        }
    }

    pub fn before_execution<E>(
        &self,
        next: S,
        mut fence: impl FnMut(&S) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let mut previous = self.execution.borrow_mut();
        if *previous != next {
            fence(&previous)?;
            *previous = next;
        }
        Ok(())
    }

    /// Try both distinct streams even if the first fails, preserving that first
    /// failure. A caller must not interpret an error as a completed release.
    pub fn synchronize<E>(
        &self,
        mut fence: impl FnMut(&S) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let execution = self.execution.borrow();
        let first = fence(&execution);
        if *execution == self.load {
            return first;
        }
        let load = fence(&self.load);
        first.and(load)
    }
}

/// A streaming, cancellable, multimodal (text + vision) text-LLM provider.
///
/// Implemented by backends (`mlx-llm`, `candle-llm`) and registered through
/// [`crate::registry`]. `&self`: a loaded provider is immutable and shareable.
///
/// Cancellation rides on [`TextLlmRequest::cancel`]: an already-cancelled request must return
/// [`Error::Canceled`](crate::Error::Canceled) before any inference; a mid-stream cancel stops
/// promptly and may return a partial output marked
/// [`FinishReason::Cancelled`](crate::FinishReason::Cancelled).
pub trait TextLlm {
    /// The provider's identity + declared capabilities (constructible without running inference).
    fn descriptor(&self) -> &TextLlmDescriptor;

    /// Return this provider's narrow StarVector extension when it implements one.
    ///
    /// Providers are always loaded through the ordinary [`TextLlm`](Self) registry.  This
    /// object-safe view lets a host retain that established identity, loading, cancellation, and
    /// error lifecycle while opting into StarVector's typed SVG boundary.  `None` is the safe
    /// default for every non-StarVector text provider.
    fn as_starvector_provider(&self) -> Option<&dyn crate::StarVectorProvider> {
        None
    }

    /// What the load produced: the requested weight format, the projection kinds the loaded
    /// model actually holds (sc-24139) and every requested optional accelerator the load did not
    /// attach (sc-24444). `None` (the default) when the provider does not report it.
    fn load_report(&self) -> Option<crate::report::LoadReport> {
        None
    }

    /// Fence this provider's OWN asynchronous execution before a caller releases
    /// its model/cache owner. Success means the work completed, NOT that dropping
    /// a borrowed handle evicts any weights or that memory was reclaimed.
    ///
    /// The default is an explicit capability refusal, never a no-op fence. A
    /// staged caller must retain ownership/report uncertainty on failure and
    /// preserve an existing primary inference/cancellation error during cleanup.
    fn synchronize_before_release(&self) -> Result<()> {
        Err(crate::Error::Unsupported(format!(
            "text provider '{}' does not implement an owned release fence; synchronization and memory reclamation cannot be claimed",
            self.descriptor().id
        )))
    }

    /// Cheap, pre-inference validation of a request against this provider's capabilities. Must
    /// reject (not silently ignore) anything outside the declared surface.
    fn validate(&self, req: &TextLlmRequest) -> Result<()>;

    /// Generate, streaming a [`StreamEvent`] per token through `on_event`, and returning the final
    /// [`TextLlmOutput`]. The terminal [`StreamEvent::Done`] carries the same finish reason + usage
    /// as the returned output.
    ///
    /// When [`TextLlmRequest::progress`] is present, install it with
    /// [`TextLlmRequest::install_progress`] while synchronous generation executes. Shared seams
    /// report actual tokenization, uncached prefill and continuation work; reused/precomputed
    /// conditioning must not synthesize a prefill phase. The guard clears the sink on every exit.
    /// Never retain a thread-bound installation across an asynchronous suspension; carry the
    /// request-owned observer to the execution thread and install it only there.
    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TextLlmOutput>;

    /// Convenience: generate without observing the stream (collect the final output only).
    fn complete(&self, req: &TextLlmRequest) -> Result<TextLlmOutput> {
        self.generate(req, &mut |_| {})
    }

    /// Generate several independent requests (sc-20681), streaming `(request index, event)`
    /// through `on_event` and returning one result per request, in request order. A backend with
    /// continuous batching decodes the requests it can batch together (each keeping its own
    /// cancellation, KV-cache policy and report); the default — and any request a backend cannot
    /// batch — runs one after another through [`Self::generate`], so every provider supports it.
    fn generate_batch(
        &self,
        reqs: &[TextLlmRequest],
        on_event: &mut dyn FnMut(usize, StreamEvent),
    ) -> Vec<Result<TextLlmOutput>> {
        reqs.iter()
            .enumerate()
            .map(|(i, req)| self.generate(req, &mut |event| on_event(i, event)))
            .collect()
    }
}

#[cfg(test)]
mod release_tests {
    use super::*;

    struct Unfenced(TextLlmDescriptor);
    impl TextLlm for Unfenced {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.0
        }
        fn validate(&self, _: &TextLlmRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _: &TextLlmRequest,
            _: &mut dyn FnMut(StreamEvent),
        ) -> Result<TextLlmOutput> {
            Err(crate::Error::Unsupported("not an inference fixture".into()))
        }
    }

    #[test]
    fn execution_fence_same_stream_is_inert_and_switch_precedes_new_work() {
        let state = OwnedExecutionFence::new(0u8, 1u8);
        let mut events = Vec::new();
        state
            .before_execution(1, |stream| {
                events.push(*stream);
                Ok::<(), &str>(())
            })
            .unwrap();
        assert!(events.is_empty());
        state
            .before_execution(2, |stream| {
                events.push(*stream);
                Ok::<(), &str>(())
            })
            .unwrap();
        events.push(99); // The caller may start new work only after before_execution succeeds.
        assert_eq!(events, vec![1, 99]);
        assert_eq!(*state.execution.borrow(), 2);
    }

    #[test]
    fn execution_fence_failed_switch_keeps_previous_stream_and_blocks_new_work() {
        let state = OwnedExecutionFence::new(0u8, 1u8);
        let mut events = Vec::new();
        let result = state.before_execution(2, |stream| {
            events.push(*stream);
            Err::<(), _>("fence failed")
        });
        if result.is_ok() {
            events.push(99);
        }
        assert_eq!(result, Err("fence failed"));
        assert_eq!(events, vec![1]);
        assert_eq!(*state.execution.borrow(), 1);
    }

    #[test]
    fn execution_fence_deduplicates_and_attempts_both_preserving_first_error() {
        let same = OwnedExecutionFence::new(1u8, 1u8);
        let mut events = Vec::new();
        same.synchronize(|stream| {
            events.push(*stream);
            Ok::<(), &str>(())
        })
        .unwrap();
        assert_eq!(events, vec![1]);
        events.clear();
        let distinct = OwnedExecutionFence::new(0u8, 1u8);
        let result = distinct.synchronize(|stream| {
            events.push(*stream);
            Err::<(), _>(if *stream == 1 { "first" } else { "second" })
        });
        assert_eq!(result, Err("first"));
        assert_eq!(events, vec![1, 0]);
    }

    #[test]
    fn execution_fence_retains_only_two_owned_handles() {
        use std::{cell::Cell, rc::Rc};
        struct Handle {
            id: u8,
            live: Rc<Cell<usize>>,
        }
        impl PartialEq for Handle {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }
        impl Drop for Handle {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }
        let live = Rc::new(Cell::new(0));
        let make = |id| {
            live.set(live.get() + 1);
            Handle {
                id,
                live: live.clone(),
            }
        };
        let state = OwnedExecutionFence::new(make(0), make(1));
        assert_eq!(live.get(), 2);
        for id in 2..20 {
            state
                .before_execution(make(id), |_| Ok::<(), &str>(()))
                .unwrap();
            assert_eq!(live.get(), 2);
        }
        drop(state);
        assert_eq!(live.get(), 0);
    }

    #[test]
    fn absent_release_fence_is_typed_unsupported_not_success() {
        let provider = Unfenced(TextLlmDescriptor {
            id: "unfenced-fixture".into(),
            family: "fixture".into(),
            backend: "fixture".into(),
            capabilities: Default::default(),
        });
        let engine: &dyn TextLlm = &provider;
        assert!(
            matches!(engine.synchronize_before_release(),Err(crate::Error::Unsupported(message)) if message.contains("owned release fence")&&message.contains("unfenced-fixture"))
        );
    }
}
