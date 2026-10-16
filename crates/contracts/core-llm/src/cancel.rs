//! Cooperative cancellation.
//!
//! Cancellation rides *in-band* on the request ([`TextLlmRequest::cancel`](crate::TextLlmRequest))
//! rather than as a separate argument. The contract: a provider handed an already-cancelled request
//! must return [`Error::Canceled`](crate::Error::Canceled) **before** running inference; a cancel
//! that trips mid-stream stops promptly, and a text LLM that has already emitted tokens may return
//! a partial [`TextLlmOutput`](crate::TextLlmOutput) marked
//! [`FinishReason::Cancelled`](crate::FinishReason::Cancelled).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A cheap, clonable, thread-safe cancellation flag shared between a caller and a provider.
#[derive(Clone, Default, Debug)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    /// A fresh, un-cancelled flag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adopt a caller-owned flag. Media-to-LLM pre-generation stages share the same atomic
    /// rather than forwarding cancellation only when a streamed token arrives (sc-26238).
    /// This lets an engine observe a cancel during prefill as well as during decoding.
    pub fn from_arc(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Clear the flag for reuse.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adopted_atomic_is_shared_before_any_stream_event() {
        let atomic = Arc::new(AtomicBool::new(false));
        let flag = CancelFlag::from_arc(atomic.clone());
        atomic.store(true, Ordering::SeqCst);
        assert!(flag.is_cancelled());
        flag.reset();
        assert!(!atomic.load(Ordering::SeqCst));
        flag.cancel();
        assert!(atomic.load(Ordering::SeqCst));
    }

    #[test]
    fn clones_share_state() {
        let a = CancelFlag::new();
        let b = a.clone();
        assert!(!a.is_cancelled());
        b.cancel();
        assert!(a.is_cancelled());
        a.reset();
        assert!(!b.is_cancelled());
    }
}
