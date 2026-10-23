//! Source-bound lifecycle registration and actual-invocation selection only.
//! None of these operations schedules native work or grants retirement credit.
//! Registered scopes are NOT active selectors until the bound executor invokes
//! a matching configured request. Native eval/fence success is separate.
use crate::{Error, Result};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeRetirementBinding {
    pub owner_token: usize,
    pub work: [u8; 32],
    pub provider: &'static str,
}
struct Configured {
    binding: NativeRetirementBinding,
    valid: Cell<bool>,
    in_use: Cell<bool>,
    invocation: Cell<u64>,
    epoch: Cell<u64>,
    request_address: Cell<usize>,
    model_entered: Cell<bool>,
}
static NEXT_INVOCATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeRetirementEpoch {
    pub invocation: u64,
    pub epoch: u64,
}
thread_local! {
    static ACTIVE:RefCell<Option<Rc<Configured>>>=const{RefCell::new(None)};
}
/// Own this only after exact provider configure_request succeeds. It keeps the
/// registration available but DOES NOT arm native retirement on this thread.
/// The Rc makes registration/leases thread-affine; no guard crosses a handoff.
pub struct ConfiguredNativeRetirement {
    inner: Rc<Configured>,
}
impl ConfiguredNativeRetirement {
    pub fn register(binding: NativeRetirementBinding) -> Self {
        let inner = Rc::new(Configured {
            binding,
            valid: Cell::new(true),
            in_use: Cell::new(false),
            invocation: Cell::new(0),
            epoch: Cell::new(0),
            request_address: Cell::new(0),
            model_entered: Cell::new(false),
        });
        Self { inner }
    }
    pub fn invalidate(&self) {
        self.inner.valid.set(false);
    }
    /// Exact lease from this registration, never a caller-controlled registry lookup.
    pub fn lease(&self) -> NativeRetirementLease {
        NativeRetirementLease {
            inner: self.inner.clone(),
        }
    }
}
impl Drop for ConfiguredNativeRetirement {
    fn drop(&mut self) {
        self.invalidate();
    }
}
#[derive(Clone)]
pub struct NativeRetirementLease {
    inner: Rc<Configured>,
}
/// Actual generation-invocation selector. Drop only restores the previous
/// selector; it never flushes, acknowledges, publishes completion or gives
/// lifetime credit. A dropped/finished outer lifecycle invalidates its lease.
pub struct NativeRetirementScope {
    previous: Option<Rc<Configured>>,
    inner: Rc<Configured>,
}
impl NativeRetirementScope {
    pub fn enter_configured(lease: &NativeRetirementLease) -> Result<Self> {
        if !lease.inner.valid.get() {
            return Err(Error::Unsupported(
                "native retirement lifecycle expired before generation".into(),
            ));
        }
        let previous = ACTIVE.with(|slot| slot.borrow().clone());
        if previous
            .as_ref()
            .is_some_and(|entry| !entry.valid.get() || !Rc::ptr_eq(entry, &lease.inner))
        {
            return Err(Error::Unsupported(
                "native retirement lifecycle does not match active generation scope".into(),
            ));
        }
        if previous.is_none() {
            if lease.inner.in_use.get() {
                return Err(Error::Unsupported(
                    "native retirement invocation cannot reenter through an Ordinary callback"
                        .into(),
                ));
            }
            let invocation = NEXT_INVOCATION
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |value| value.checked_add(1),
                )
                .map_err(|_| {
                    Error::Unsupported("native retirement invocation identity exhausted".into())
                })?;
            lease.inner.invocation.set(invocation);
            lease.inner.epoch.set(0);
            lease.inner.request_address.set(0);
            lease.inner.model_entered.set(false);
            lease.inner.in_use.set(true);
        }
        ACTIVE.with(|slot| *slot.borrow_mut() = Some(lease.inner.clone()));
        Ok(Self {
            previous,
            inner: lease.inner.clone(),
        })
    }
}
impl Drop for NativeRetirementScope {
    fn drop(&mut self) {
        if self.previous.is_none() {
            self.inner.in_use.set(false);
        }
        ACTIVE.with(|slot| {
            let matches = slot
                .borrow()
                .as_ref()
                .is_some_and(|active| Rc::ptr_eq(active, &self.inner));
            *slot.borrow_mut() = if matches { self.previous.take() } else { None };
        });
    }
}
/// Suppress inherited selection for a distinct Ordinary invocation, including
/// a nested generation called from a progress callback. Selector restoration
/// only: no synchronization, cleanup, acknowledgement or lifetime credit.
pub struct OrdinaryNativeRetirementScope {
    previous: Option<Rc<Configured>>,
}
impl OrdinaryNativeRetirementScope {
    pub fn enter() -> Self {
        Self {
            previous: ACTIVE.with(|slot| slot.borrow_mut().take()),
        }
    }
}
impl Drop for OrdinaryNativeRetirementScope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            let mut active = slot.borrow_mut();
            *active = if active.is_none() {
                self.previous.take()
            } else {
                None
            };
        });
    }
}
/// Scalar identities only. This records source chronology, NEVER native success.
/// Bind the ORIGINAL borrowed invocation carrier, not a deep clone or replay.
/// Scalar address is process-local only and is never serialized as evidence.
pub fn bind_native_retirement_request(address: usize) -> Result<()> {
    validate_native_retirement_scope()?;
    ACTIVE.with(|slot| {
        let active = slot.borrow();
        let entry = active.as_ref().ok_or_else(|| {
            Error::Unsupported("native invocation missing before original request binding".into())
        })?;
        if address == 0 || entry.request_address.get() != 0 {
            return Err(Error::Unsupported(
                "native invocation request already bound or invalid".into(),
            ));
        }
        entry.request_address.set(address);
        Ok(())
    })
}
pub fn native_retirement_request_matches(address: usize) -> Result<bool> {
    validate_native_retirement_scope()?;
    Ok(ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|entry| address != 0 && entry.request_address.get() == address)
    }))
}
pub struct NativeRetirementModelScope {
    inner: Rc<Configured>,
}
pub fn enter_native_retirement_model() -> Result<NativeRetirementModelScope> {
    validate_native_retirement_scope()?;
    ACTIVE.with(|slot| {
        let active = slot.borrow();
        let entry = active
            .as_ref()
            .ok_or_else(|| Error::Unsupported("native model entered without invocation".into()))?;
        if entry.request_address.get() == 0 || entry.model_entered.replace(true) {
            return Err(Error::Unsupported(
                "native SANA generation cannot reenter the same invocation".into(),
            ));
        }
        Ok(NativeRetirementModelScope {
            inner: entry.clone(),
        })
    })
}
impl Drop for NativeRetirementModelScope {
    fn drop(&mut self) {
        self.inner.model_entered.set(false);
    }
}

pub(super) fn retirement_chronology() -> Option<NativeRetirementEpoch> {
    ACTIVE.with(|slot| {
        slot.borrow().as_ref().map(|entry| NativeRetirementEpoch {
            invocation: entry.invocation.get(),
            epoch: entry.epoch.get(),
        })
    })
}
pub fn native_retirement_epoch() -> Result<Option<NativeRetirementEpoch>> {
    validate_native_retirement_scope()?;
    Ok(retirement_chronology())
}
pub fn begin_native_retirement_epoch() -> Result<Option<NativeRetirementEpoch>> {
    validate_native_retirement_scope()?;
    ACTIVE.with(|slot| {
        let active = slot.borrow();
        let Some(entry) = active.as_ref() else {
            return Ok(None);
        };
        let epoch = entry
            .epoch
            .get()
            .checked_add(1)
            .ok_or_else(|| Error::Unsupported("native retirement epoch exhausted".into()))?;
        entry.epoch.set(epoch);
        Ok(Some(NativeRetirementEpoch {
            invocation: entry.invocation.get(),
            epoch,
        }))
    })
}
pub fn native_retirement_binding() -> Option<NativeRetirementBinding> {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|entry| entry.valid.get())
            .map(|entry| entry.binding)
    })
}
/// Check before any builder/fence: an invalidated active lifecycle must fail
/// closed, never silently demote its optimistically admitted work to Ordinary.
pub fn validate_native_retirement_scope() -> Result<()> {
    ACTIVE.with(|slot| {
        if slot
            .borrow()
            .as_ref()
            .is_some_and(|entry| !entry.valid.get())
        {
            Err(Error::Unsupported(
                "native retirement lifecycle ended during generation".into(),
            ))
        } else {
            Ok(())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn binding(owner: usize) -> NativeRetirementBinding {
        NativeRetirementBinding {
            owner_token: owner,
            work: [7; 32],
            provider: "sana_1600m",
        }
    }
    #[test]
    fn mere_configured_scope_does_not_select_or_schedule_retirement() {
        assert_eq!(native_retirement_binding(), None);
        let registration = ConfiguredNativeRetirement::register(binding(1));
        let lease = registration.lease();
        assert_eq!(native_retirement_binding(), None);
        drop(registration);
        assert!(NativeRetirementScope::enter_configured(&lease).is_err());
    }
    #[test]
    fn matching_actual_invocation_is_nest_safe_and_restores_only_selector() {
        let registration = ConfiguredNativeRetirement::register(binding(2));
        let lease = registration.lease();
        let outer = NativeRetirementScope::enter_configured(&lease).unwrap();
        assert_eq!(native_retirement_binding(), Some(binding(2)));
        {
            let _inner = NativeRetirementScope::enter_configured(&lease).unwrap();
            assert_eq!(native_retirement_binding(), Some(binding(2)));
        }
        assert_eq!(native_retirement_binding(), Some(binding(2)));
        drop(outer);
        assert_eq!(native_retirement_binding(), None);
    }
    #[test]
    fn wrong_binding_cannot_inherit_active_invocation() {
        let a = ConfiguredNativeRetirement::register(binding(3));
        let b = ConfiguredNativeRetirement::register(binding(4));
        let lease = a.lease();
        let _scope = NativeRetirementScope::enter_configured(&lease).unwrap();
        let wrong = b.lease();
        assert!(NativeRetirementScope::enter_configured(&wrong).is_err());
        assert_eq!(native_retirement_binding(), Some(binding(3)));
    }
    #[test]
    fn finished_lifecycle_cannot_fall_back_to_ordinary_inside_active_work() {
        let registration = ConfiguredNativeRetirement::register(binding(5));
        let lease = registration.lease();
        let scope = NativeRetirementScope::enter_configured(&lease).unwrap();
        drop(registration);
        assert!(validate_native_retirement_scope().is_err());
        assert_eq!(native_retirement_binding(), None);
        drop(scope);
        validate_native_retirement_scope().unwrap();
    }
    #[test]
    fn ordinary_nested_invocation_suspends_and_restores_without_credit() {
        let registration = ConfiguredNativeRetirement::register(binding(7));
        let _active = NativeRetirementScope::enter_configured(&registration.lease()).unwrap();
        {
            let _ordinary = OrdinaryNativeRetirementScope::enter();
            assert_eq!(native_retirement_binding(), None);
            {
                let _nested = OrdinaryNativeRetirementScope::enter();
                assert_eq!(native_retirement_binding(), None);
            }
            assert_eq!(native_retirement_binding(), None);
        }
        assert_eq!(native_retirement_binding(), Some(binding(7)));
    }
    #[test]
    fn epochs_are_actual_invocation_local_and_not_lifetime_credit() {
        let registration = ConfiguredNativeRetirement::register(binding(8));
        let lease = registration.lease();
        let scope = NativeRetirementScope::enter_configured(&lease).unwrap();
        let first = native_retirement_epoch().unwrap().unwrap();
        assert!(first.invocation > 0);
        assert_eq!(first.epoch, 0);
        assert_eq!(begin_native_retirement_epoch().unwrap().unwrap().epoch, 1);
        {
            let _nested = NativeRetirementScope::enter_configured(&lease).unwrap();
            assert_eq!(begin_native_retirement_epoch().unwrap().unwrap().epoch, 2);
        }
        drop(scope);
        let _next = NativeRetirementScope::enter_configured(&lease).unwrap();
        let next = native_retirement_epoch().unwrap().unwrap();
        assert!(next.invocation > first.invocation);
        assert_eq!(next.epoch, 0);
    }
    #[test]
    fn ordinary_callback_cannot_reenter_the_suspended_same_invocation() {
        let registration = ConfiguredNativeRetirement::register(binding(9));
        let lease = registration.lease();
        let _active = NativeRetirementScope::enter_configured(&lease).unwrap();
        let first = begin_native_retirement_epoch().unwrap();
        {
            let _ordinary = OrdinaryNativeRetirementScope::enter();
            assert!(NativeRetirementScope::enter_configured(&lease).is_err());
        }
        assert_eq!(native_retirement_epoch().unwrap(), first);
    }
    #[test]
    fn original_request_binding_and_model_entry_are_not_inferred_from_scope_presence() {
        let registration = ConfiguredNativeRetirement::register(binding(10));
        let _active = NativeRetirementScope::enter_configured(&registration.lease()).unwrap();
        assert!(!native_retirement_request_matches(123).unwrap());
        bind_native_retirement_request(123).unwrap();
        assert!(native_retirement_request_matches(123).unwrap());
        assert!(!native_retirement_request_matches(456).unwrap());
        assert!(bind_native_retirement_request(456).is_err());
        let model = enter_native_retirement_model().unwrap();
        assert!(enter_native_retirement_model().is_err());
        {
            let _ordinary = OrdinaryNativeRetirementScope::enter();
            assert!(!native_retirement_request_matches(123).unwrap());
        }
        drop(model);
        assert!(enter_native_retirement_model().is_ok());
    }
    #[test]
    fn registration_and_selector_are_thread_local_not_handoff_tokens() {
        let registration = ConfiguredNativeRetirement::register(binding(6));
        let _active = NativeRetirementScope::enter_configured(&registration.lease()).unwrap();
        std::thread::spawn(|| {
            assert_eq!(native_retirement_binding(), None);
        })
        .join()
        .unwrap();
    }
}
