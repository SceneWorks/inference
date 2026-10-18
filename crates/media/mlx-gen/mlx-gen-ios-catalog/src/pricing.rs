//! The providers' counted iOS pricing and quantum-phase attribution, by provider id: the
//! architecture facts the anchor law ([`crate::anchor`]) scales a measured anchor by.
//!
//! A provider with no pricing entry here is still admitted at its anchors' OWN keys (an exact
//! measurement needs no scaling), but nothing of it is derivable: every other configuration is
//! refused `no_evidence`.

use mlx_gen::ios_pricing::{IosPhase, IosPricer, IosQuantumPhase};

#[cfg(test)]
thread_local! {
    static TEST_PRICING: std::cell::Cell<Option<(IosPricer, IosQuantumPhase)>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with every provider priced by `pricing` (crate tests only).
#[cfg(test)]
pub(crate) fn with_test_pricing<T>(
    pricing: (IosPricer, IosQuantumPhase),
    body: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<(IosPricer, IosQuantumPhase)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_PRICING.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(TEST_PRICING.with(|slot| slot.replace(Some(pricing))));
    body()
}

/// `(pricing, quantum-label → phase)` of a registered iOS provider.
pub fn provider_pricing(provider: &str) -> Option<(IosPricer, IosQuantumPhase)> {
    #[cfg(test)]
    if let Some(pricing) = TEST_PRICING.with(std::cell::Cell::get) {
        return Some(pricing);
    }
    let _ = provider;
    None
}

/// The phase a quantum label of `provider` runs in.
pub fn quantum_phase(provider: &str, label: &str) -> Option<IosPhase> {
    provider_pricing(provider).and_then(|(_, phase)| phase(label))
}
