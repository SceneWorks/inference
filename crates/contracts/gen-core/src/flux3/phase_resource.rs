//! Actual model-local ownership for an optional exclusive phase (FLUX3 PUP).
//!
//! The slot removes/drops its OWN owner only after a successful owned fence. A
//! failed/unsupported fence retains that owner and is never credited as eviction.
//! No byte credit is returned: external aliases/allocator reservations must still
//! be accounted by the caller. Partial-load cleanup belongs to the load callback.
use crate::{CancelFlag, Error, Result};
use std::sync::{Mutex, MutexGuard};

struct State<T> {
    resource: Option<T>,
    pending_release: bool,
}

pub struct PhaseResourceSlot<T> {
    state: Mutex<State<T>>,
}
impl<T> Default for PhaseResourceSlot<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                resource: None,
                pending_release: false,
            }),
        }
    }
}

impl<T> PhaseResourceSlot<T> {
    /// The ownership lock. It is held across load and run, so a panic there poisons it. A
    /// poisoned slot is recovered rather than bricked: any owner it still holds is marked
    /// `pending_release`, so the next request fences and drops it before anything reuses it
    /// (the panicking phase may have left that owner mid-operation).
    fn lock(&self) -> Result<MutexGuard<'_, State<T>>> {
        Ok(match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.state.clear_poison();
                let mut guard = poisoned.into_inner();
                guard.pending_release = guard.resource.is_some();
                guard
            }
        })
    }

    /// Whether the actual owner is retained, not a prediction of physical bytes.
    pub fn is_loaded(&self) -> Result<bool> {
        Ok(self.lock()?.resource.is_some())
    }

    /// A prior cleanup attempt failed. An outer error handler can report/retain
    /// this state instead of silently retrying the same fence twice in one phase.
    pub fn cleanup_pending(&self) -> Result<bool> {
        Ok(self.lock()?.pending_release)
    }

    /// Explicit cache eviction, never merely dropping a borrowed local handle.
    /// On fence failure the slot retains ownership, including for later retries.
    pub fn release_cached(
        &self,
        mut fence: impl FnMut(&T) -> Result<()>,
        mut after_drop: impl FnMut(),
    ) -> Result<()> {
        let mut state = self.lock()?;
        Self::release_locked(&mut state, &mut fence, &mut after_drop)
    }

    fn release_locked(
        state: &mut State<T>,
        fence: &mut impl FnMut(&T) -> Result<()>,
        after_drop: &mut impl FnMut(),
    ) -> Result<()> {
        if let Some(resource) = state.resource.as_ref() {
            if let Err(error) = fence(resource) {
                state.pending_release = true;
                return Err(error);
            }
            drop(state.resource.take());
            state.pending_release = false;
            after_drop();
        }
        Ok(())
    }

    fn finish<Out>(
        state: &mut State<T>,
        primary: Result<Out>,
        fence: &mut impl FnMut(&T) -> Result<()>,
        after_drop: &mut impl FnMut(),
    ) -> Result<Out> {
        match Self::release_locked(state, fence, after_drop) {
            Ok(()) => primary,
            Err(cleanup) if primary.is_err() => {
                eprintln!("FLUX3 phase cleanup failed; original error/cancellation preserved, owner retained, no reclamation claimed: {cleanup}");
                primary
            }
            Err(cleanup) => Err(cleanup),
        }
    }

    /// `needed=false` never calls `load` and evicts any prior cached owner before
    /// bypass. `exclusive=true` evicts a prior resident owner before loading the
    /// bounded instance, then fences/drops it BEFORE the caller's next phase.
    /// Resident success keeps its owner and adds no unnecessary release barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn with_request<Out>(
        &self,
        needed: bool,
        exclusive: bool,
        cancel: &CancelFlag,
        load: impl FnOnce() -> Result<T>,
        run: impl FnOnce(Option<&T>) -> Result<Out>,
        mut fence: impl FnMut(&T) -> Result<()>,
        mut after_drop: impl FnMut(),
    ) -> Result<Out> {
        let mut state = self.lock()?;
        if cancel.is_cancelled() {
            return Self::finish(
                &mut state,
                Err(Error::Canceled),
                &mut fence,
                &mut after_drop,
            );
        }
        if !needed || exclusive || state.pending_release {
            if let Err(cleanup) = Self::release_locked(&mut state, &mut fence, &mut after_drop) {
                if cancel.is_cancelled() {
                    eprintln!("FLUX3 phase cleanup failed during cancellation; owner retained, no reclamation claimed: {cleanup}");
                    return Err(Error::Canceled);
                }
                return Err(cleanup);
            }
        }
        if cancel.is_cancelled() {
            return Self::finish(
                &mut state,
                Err(Error::Canceled),
                &mut fence,
                &mut after_drop,
            );
        }
        if !needed {
            return run(None);
        }
        if state.resource.is_none() {
            // A failing loader owns/fences its partial state; there is no object
            // here on which this slot could truthfully claim a release fence.
            state.resource = Some(load()?);
        }
        let primary = if cancel.is_cancelled() {
            Err(Error::Canceled)
        } else {
            run(state.resource.as_ref())
        };
        let primary = if primary.is_ok() && cancel.is_cancelled() {
            Err(Error::Canceled)
        } else {
            primary
        };
        if exclusive || primary.is_err() {
            Self::finish(&mut state, primary, &mut fence, &mut after_drop)
        } else {
            primary
        }
    }
}

/// Publish the exact conditioning/denoising boundary only after the provider's materialization
/// AND device fence succeed. Cancellation before or during that work suppresses the event; a
/// consumer canceling in the callback prevents the first denoiser forward. Positive `Step`
/// events stay 1-based; this boundary emits neither a step nor a terminal event.
pub fn conditioning_complete(
    cancel: &CancelFlag,
    materialize_and_synchronize: impl FnOnce() -> Result<()>,
    progress: &mut dyn FnMut(crate::Progress),
) -> Result<()> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    materialize_and_synchronize()?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    progress(crate::Progress::ConditioningComplete);
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    Ok(())
}

#[cfg(test)]
#[path = "phase_resource_tests.rs"]
mod tests;
