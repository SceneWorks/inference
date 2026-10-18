//! The providers' counted iOS pricing and quantum-phase attribution, by provider id: the
//! architecture facts the anchor law ([`crate::anchor`]) scales a measured anchor by.
//!
//! Every registered iOS provider has an entry (`every_registered_provider_is_priced`). A provider
//! without one would still be admitted at its anchors' OWN keys (an exact measurement needs no
//! scaling), but nothing of it would be derivable: every other configuration refuses `no_evidence`.

use mlx_gen::ios_pricing::{IosPhase, IosPricer, IosQuantumPhase};

type Pricing = (IosPricer, IosQuantumPhase);

#[cfg(test)]
thread_local! {
    /// `Some(p)`: every provider priced by `p` (or by nothing, for `Some(None)`).
    static TEST_PRICING: std::cell::Cell<Option<Option<Pricing>>> =
        const { std::cell::Cell::new(None) };
}

/// Run `body` with every provider priced by `pricing` — `None`: no provider priced (crate tests
/// only).
#[cfg(test)]
pub(crate) fn with_test_pricing<T>(pricing: Option<Pricing>, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Option<Pricing>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_PRICING.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(TEST_PRICING.with(|slot| slot.replace(Some(pricing))));
    body()
}

/// `(pricing, quantum-label → phase)` of a registered iOS provider.
pub fn provider_pricing(provider: &str) -> Option<Pricing> {
    #[cfg(test)]
    if let Some(pricing) = TEST_PRICING.with(std::cell::Cell::get) {
        return pricing;
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
        crate::sdxl::SDXL_ID => Some((
            mlx_gen_sdxl::ios_pricing::ios_pricing,
            mlx_gen_sdxl::ios_pricing::ios_quantum_phase,
        )),
        id if crate::zimage::is_z_image(id) => Some((
            mlx_gen_z_image::ios_pricing::ios_pricing,
            mlx_gen_z_image::ios_pricing::ios_quantum_phase,
        )),
        _ => None,
    }
}

/// The anchor VARIANT of a `(provider, mode)`: `base` when the provider has proven the mode
/// component-additive on its CPU fixtures (`ios_anchor_law::measure_phases` + `derive`: the measured
/// mode peak never exceeds the one derived from the text-mode run), so it is derived from the text
/// anchor of its architecture; otherwise the mode key itself — a separate anchor family.
pub fn anchor_variant(provider: &str, mode: &str) -> String {
    let non_additive: &[&str] = match provider {
        _ => &[],
    };
    if non_additive.contains(&mode) {
        mode.to_string()
    } else {
        "base".to_string()
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

/// HOST INTEGRATION CHECK over real snapshot HEADERS (no tensor payload, no graph, no device): the
/// catalog builds each provider's fully optimized candidate exactly as admission does
/// (`build_candidate` at the deepest implemented rung's cheapest selection) and prices it through
/// this dispatch — the request shape the anchor law hands the providers' pricing. Prints the
/// whole-run counted peak + the 512 MiB formula reserve against 6142 MiB at 512/768/1024.
/// `IOS_PRICING_SNAPSHOTS=<dir>` holds `<route>/<tier>/` snapshot directories (symlinks allowed);
/// a missing one is skipped. Run with `--ignored --nocapture`.
#[cfg(test)]
mod host_integration {
    use crate::ios_admission::{build_candidate, IosWork};
    use mlx_gen::gen_core::{
        GenerationRequest, MemoryNumericTier, MemoryStrategy, MemoryStrategySupport, Precision,
        Quant, WeightsSource,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;

    #[test]
    #[ignore = "reads provisioned snapshot headers (IOS_PRICING_SNAPSHOTS)"]
    fn fully_optimized_candidates_price_through_the_catalog_dispatch() {
        let Some(root) = std::env::var_os("IOS_PRICING_SNAPSHOTS").map(std::path::PathBuf::from)
        else {
            println!("IOS_PRICING_SNAPSHOTS unset: nothing to price");
            return;
        };
        let registry = crate::provider_registry().unwrap();
        const MIB: u64 = 1 << 20;
        println!(
            "| provider | route | tier | edge | rung / selection | priced peak + 512 MiB | binding phase | vs 6142 |"
        );
        println!("|---|---|---|---:|---|---:|---|---|");
        let mut checked = 0;
        for (provider, route) in [
            ("sana_1600m", "sana_1600m"),
            ("sana_sprint_1600m", "sana_sprint_1600m"),
            ("z_image_turbo", "z_image_turbo"),
            ("sdxl", "sdxl"),
            ("sdxl", "realvisxl"),
            ("sdxl", "illustrious_xl_v2"),
        ] {
            for (tier, quant) in [
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
                ("bf16", None),
            ] {
                let dir = root.join(route).join(tier);
                if !dir.is_dir() {
                    continue;
                }
                let source = WeightsSource::Dir(dir);
                let mut spec = match provider {
                    "sdxl" => crate::sdxl::sdxl_load_spec(route, source).unwrap(),
                    id if crate::zimage::is_z_image(id) => {
                        crate::zimage::z_image_turbo_load_spec(source, None)
                    }
                    _ => crate::sana_load_spec(source),
                };
                spec.precision = Precision::Bf16;
                spec.quantize = quant;
                let contract = registry
                    .memory_strategy_contract(provider, &spec)
                    .unwrap()
                    .unwrap();
                let numeric = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                };
                let (strategy, selection) = MemoryStrategy::ALL
                    .into_iter()
                    .rev()
                    .filter(|s| {
                        contract.capability(*s).is_some_and(|c| {
                            matches!(c.support, MemoryStrategySupport::Implemented)
                        })
                    })
                    .find_map(|s| {
                        let space = contract.selection_space(s, numeric, false).ok()?;
                        space.last().copied().map(|sel| (s, sel))
                    })
                    .unwrap();
                for edge in [512u32, 768, 1024] {
                    let work = IosWork::Generate(GenerationRequest {
                        prompt: "A red ceramic teapot on a wooden table in daylight".into(),
                        width: edge,
                        height: edge,
                        count: 1,
                        ..Default::default()
                    });
                    let c = build_candidate(
                        &registry,
                        provider,
                        spec.clone(),
                        Some(selection),
                        strategy,
                        work,
                    )
                    .unwrap_or_else(|e| panic!("{provider}/{route}/{tier}/{edge}: {e}"));
                    let (pricer, _) = super::provider_pricing(provider).unwrap();
                    let window = crate::ios_admission::token_window(&c).unwrap();
                    let priced = pricer(&IosPricingRequest {
                        provider,
                        spec: &c.spec,
                        mode: &c.context.mode,
                        geometry: c.context.geometry,
                        selection: &c.context.selection,
                        execution: &c.execution,
                        token_window: window,
                    })
                    .unwrap_or_else(|e| panic!("{provider}/{route}/{tier}/{edge}: {e}"));
                    let worst = priced.phases.iter().max_by_key(|p| p.peak_bytes()).unwrap();
                    let total = priced.peak_bytes() + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;
                    println!(
                        "| {provider} | {route} | {tier} | {edge} | {:?} {:?} | {} | {} ({}) | {} |",
                        strategy,
                        selection.parameters,
                        total.div_ceil(MIB),
                        worst.phase.as_key(),
                        worst.binding,
                        if total <= 6142 * MIB { "fits" } else { "OVER" }
                    );
                    assert!(
                        total <= 6142 * MIB,
                        "{provider}/{route}/{tier}/{edge} does not fit"
                    );
                    checked += 1;
                }
            }
        }
        println!("{checked} fully optimized cells priced through the catalog dispatch");
    }
}
