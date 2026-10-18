//! The providers' counted iOS pricing and quantum-phase attribution, by provider id: the
//! architecture facts the anchor law ([`crate::anchor`]) scales a measured anchor by.
//!
//! Every registered iOS provider has an entry (`every_registered_provider_is_priced`). A provider
//! without one would still be admitted at its anchors' OWN keys (an exact measurement needs no
//! scaling), but nothing of it would be derivable: every other configuration refuses `no_evidence`.

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
    match provider {
        "sana_1600m" | "sana_sprint_1600m" => Some((
            mlx_gen_sana::ios_pricing::ios_pricing,
            mlx_gen_sana::ios_pricing::ios_quantum_phase,
        )),
        crate::anima::ANIMA_TURBO_ID => Some((
            mlx_gen_anima::ios_pricing::ios_pricing,
            mlx_gen_anima::ios_pricing::ios_quantum_phase,
        )),
        "iris_3b" | crate::ios_admission::IRIS_RESTORE_ID | crate::ios_admission::IRIS_DEPTH_ID => {
            Some((
                mlx_gen_iris::ios_pricing::ios_pricing,
                mlx_gen_iris::ios_pricing::ios_quantum_phase,
            ))
        }
        id if crate::zimage::is_z_image(id) => Some((
            mlx_gen_z_image::ios_pricing::ios_pricing,
            mlx_gen_z_image::ios_pricing::ios_quantum_phase,
        )),
        _ => None,
    }
}

/// The phase a quantum label of `provider` runs in.
pub fn quantum_phase(provider: &str, label: &str) -> Option<IosPhase> {
    provider_pricing(provider).and_then(|(_, phase)| phase(label))
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_registered_provider_is_priced() {
        let registry = crate::provider_registry().unwrap();
        for registration in registry.memory_strategy_registrations() {
            assert!(
                super::provider_pricing(registration.provider_id).is_some(),
                "{} has no counted iOS pricing",
                registration.provider_id
            );
        }
    }
}
