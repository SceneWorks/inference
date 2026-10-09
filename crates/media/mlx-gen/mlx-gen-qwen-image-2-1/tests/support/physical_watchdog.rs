//! Tensor-free admission for the physical-memory diagnostic; all values are bytes.

#[derive(Clone, Copy, Debug)]
pub struct Host {
    pub total: u64,
    pub available: u64,
    pub recommended: u64,
    pub mlx_limit: u64,
    pub pressure: u64,
    pub baseline_physical: u64,
    pub baseline_active: u64,
    pub baseline_cache: u64,
    pub cache_limit: u64,
}

/// The complete tensor-free physical admission decision. `effective_cache_limit`
/// is the most allocator cache this task may retain while keeping
/// `full_envelope` at or below `physical_ceiling`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionGrant {
    pub physical_ceiling: u64,
    pub nonallocator_overhead: u64,
    pub active_envelope: u64,
    pub requested_cache_limit: u64,
    pub effective_cache_limit: u64,
    pub full_envelope: u64,
}

/// The small allocator surface needed by [`ScopedCacheGrant`]. Keeping this
/// tensor-free lets the ordinary CPU suite prove scope and restoration.
pub trait CacheAllocator {
    fn set_cache_limit(&self, limit: usize) -> usize;
    fn clear_cache(&self);
}

/// A task-owned cache limit which restores the caller's setting on every exit,
/// including `Result::Err` and unwinding panics.
pub struct ScopedCacheGrant<A: CacheAllocator> {
    allocator: A,
    grant: AdmissionGrant,
    previous: usize,
    effective: usize,
}

impl<A: CacheAllocator> ScopedCacheGrant<A> {
    pub fn enter(allocator: A, grant: AdmissionGrant) -> Result<Self, String> {
        let requested = usize::try_from(grant.effective_cache_limit).map_err(|_| {
            format!(
                "cache grant {} does not fit allocator usize",
                grant.effective_cache_limit
            )
        })?;
        // Tighten to zero while discovering the inherited limit, so a caller
        // which was already stricter is never raised even for one API call.
        let previous = allocator.set_cache_limit(0);
        let effective = previous.min(requested);
        let zero = allocator.set_cache_limit(effective);
        if zero != 0 {
            allocator.set_cache_limit(previous);
            return Err(format!(
                "allocator zero clamp read-back mismatch: expected 0, observed {zero}"
            ));
        }
        let observed = allocator.set_cache_limit(0);
        if observed != effective {
            allocator.set_cache_limit(previous);
            return Err(format!(
                "allocator cache grant read-back mismatch: requested {effective}, observed {observed}"
            ));
        }
        let zero = allocator.set_cache_limit(effective);
        if zero != 0 {
            allocator.set_cache_limit(previous);
            return Err(format!(
                "allocator final cache grant install mismatch: expected previous 0, observed {zero}"
            ));
        }
        let guard = Self {
            allocator,
            grant,
            previous,
            effective,
        };
        guard.allocator.clear_cache();
        Ok(guard)
    }

    pub fn previous(&self) -> usize {
        self.previous
    }

    pub fn effective(&self) -> usize {
        self.effective
    }

    pub fn grant(&self) -> AdmissionGrant {
        self.grant
    }
}

impl<A: CacheAllocator> Drop for ScopedCacheGrant<A> {
    fn drop(&mut self) {
        self.allocator.set_cache_limit(self.previous);
    }
}

/// Reserve 15% of physical RAM and 5% of currently reclaimable pages, and keep
/// the engine's 15% MLX-limit headroom. The recommended Metal working set also bounds allocator residency;
/// measured non-allocator baseline bytes are allowed separately. The sampler
/// still aborts if cache retention consumes this budget before a step finishes.
pub fn admit(
    host: Host,
    envelope: u64,
    explicit_cap: Option<u64>,
) -> Result<AdmissionGrant, String> {
    if host.total == 0
        || host.available == 0
        || host.available > host.total
        || host.recommended == 0
        || host.mlx_limit == 0
        || envelope == 0
    {
        return Err("host/available/Metal memory census is incomplete".into());
    }
    if host.pressure != 1 {
        return Err(format!(
            "host memory pressure is not normal: {}",
            host.pressure
        ));
    }
    let allocator_baseline = host
        .baseline_active
        .checked_add(host.baseline_cache)
        .ok_or("active plus cache baseline overflows u64")?;
    let overhead = host.baseline_physical.saturating_sub(allocator_baseline);
    let fraction = |bytes: u64, numerator: u64| (bytes / 100) * numerator;
    let available_cap = host
        .baseline_physical
        .checked_add(fraction(host.available, 95))
        .ok_or("reclaimable physical cap overflows u64")?;
    let recommended_cap = host
        .recommended
        .checked_add(overhead)
        .ok_or("recommended physical cap overflows u64")?;
    let mlx_cap = fraction(host.mlx_limit, 85)
        .checked_add(overhead)
        .ok_or("MLX physical cap overflows u64")?;
    let safe_cap = fraction(host.total, 85)
        .min(available_cap)
        .min(recommended_cap)
        .min(mlx_cap)
        .min(explicit_cap.unwrap_or(u64::MAX));
    let required = envelope
        .checked_add(overhead)
        .ok_or("active envelope plus measured overhead overflows u64")?;
    if required > safe_cap {
        return Err(format!("selected preflight {envelope} + measured overhead {overhead} requires {required} bytes; verified physical cap is {safe_cap} bytes"));
    }
    let requested_full_envelope = required
        .checked_add(host.cache_limit)
        .ok_or("active envelope plus requested cache overflows u64")?;
    // Preserve the prior admission ceiling whenever it was already lower than
    // the host cap; the repair can only tighten the cache which consumes it.
    let physical_ceiling = requested_full_envelope.min(safe_cap);
    let effective_cache_limit = host.cache_limit.min(
        physical_ceiling
            .checked_sub(required)
            .ok_or("verified cap is below the required active envelope")?,
    );
    let full_envelope = required
        .checked_add(effective_cache_limit)
        .ok_or("active envelope plus cache grant overflows u64")?;
    Ok(AdmissionGrant {
        physical_ceiling,
        nonallocator_overhead: overhead,
        active_envelope: envelope,
        requested_cache_limit: host.cache_limit,
        effective_cache_limit,
        full_envelope,
    })
}

/// The test-only materialized diagnostic reserves its full live plus free-cache
/// envelope before constructing any tensor. Ordinary training admission is unchanged.
pub fn admit_full(host: Host, envelope: u64, explicit_cap: Option<u64>) -> Result<u64, String> {
    let grant = admit(host, envelope, explicit_cap)?;
    if grant.effective_cache_limit != host.cache_limit {
        let required = grant
            .active_envelope
            .checked_add(grant.nonallocator_overhead)
            .and_then(|v| v.checked_add(host.cache_limit))
            .ok_or("numeric full physical envelope overflows u64")?;
        return Err(format!(
            "numeric full physical envelope {required} exceeds verified cap {}; active-only admission is insufficient",
            grant.physical_ceiling
        ));
    }
    Ok(grant.physical_ceiling)
}

/// Reservation is independent of the allocator's actual cache cap. This is an
/// estimator only; it neither raises that cap nor changes ordinary admission.
pub fn admit_numeric_full(
    mut host: Host,
    envelope: u64,
    frozen_free_cache: u64,
    explicit_cap: Option<u64>,
) -> Result<u64, String> {
    host.cache_limit = frozen_free_cache;
    admit_full(host, envelope, explicit_cap)
}

/// Derive a diagnostic-only cache grant from the frozen allowance and current
/// host headroom. The active envelope and every host reserve stay unchanged;
/// only freed-buffer retention may be clamped. The typed allocator guard still
/// refuses a mismatched read-back and restores the caller's policy on exit.
pub fn admit_numeric_scoped(
    host: Host,
    envelope: u64,
    frozen_free_cache: u64,
    explicit_cap: Option<u64>,
) -> Result<(u64, AdmissionGrant), String> {
    if frozen_free_cache == 0 {
        return Err("numeric diagnostic free-cache allowance is zero".into());
    }
    let mut scoped_host = host;
    // Price the frozen request, rather than the process's inherited setting.
    // ScopedCacheGrant independently refuses to raise a tighter inherited cap.
    scoped_host.cache_limit = frozen_free_cache;
    let grant = admit(scoped_host, envelope, explicit_cap)?;
    Ok((grant.physical_ceiling, grant))
}

/// Parse the printed `vm_stat` snapshot, whose "Pages free" excludes speculative
/// pages. Its printed free, speculative and inactive buckets are disjoint.
/// Raw Mach `free_count` already contains speculative pages and is a different
/// format: never add speculative again to that raw counter. Purgeable and
/// file-backed counters can overlap and are excluded from this sum.
/// <https://github.com/apple-oss-distributions/system_cmds/blob/main/vm_stat/vm_stat.c>
pub fn reclaimable_bytes(text: &str) -> Result<u64, String> {
    let page = text
        .split("page size of ")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok())
        .filter(|page| *page > 0)
        .ok_or("missing vm_stat page size")?;
    let pages = |label: &str| {
        text.lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix(label)
                    .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
            })
            .ok_or_else(|| format!("missing vm_stat {label}"))
    };
    Ok(pages("Pages free:")?
        .saturating_add(pages("Pages speculative:")?)
        .saturating_add(pages("Pages inactive:")?)
        .saturating_mul(page))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    const GIB: u64 = 1 << 30;
    fn host() -> Host {
        Host {
            total: 128 * GIB,
            available: 120 * GIB,
            recommended: 104 * GIB,
            mlx_limit: 156 * GIB,
            pressure: 1,
            baseline_physical: GIB,
            baseline_active: GIB / 4,
            baseline_cache: GIB / 4,
            cache_limit: 80 * GIB,
        }
    }
    #[test]
    fn selected_edit_is_admitted_only_with_verified_headroom() {
        let envelope = 100_983_708_416;
        let grant = admit(host(), envelope, None).unwrap();
        assert!(grant.physical_ceiling >= envelope + GIB / 2);
        assert_eq!(grant.physical_ceiling, 104 * GIB + GIB / 2);
        assert!(grant.physical_ceiling < 128 * GIB * 85 / 100);
        assert_eq!(grant.full_envelope, grant.physical_ceiling);
        assert_eq!(grant.effective_cache_limit, 10_685_441_280);
        assert!(admit(host(), envelope, Some(100_000_000_000)).is_err());
    }

    #[test]
    fn failed_native_case_grants_only_the_cache_that_fits_the_full_envelope() {
        let measured = Host {
            total: 137_438_953_472,
            available: 106_946_822_144,
            recommended: 115_448_725_504,
            mlx_limit: 130_567_005_798,
            pressure: 1,
            baseline_physical: 116_310_640,
            baseline_active: 0,
            baseline_cache: 0,
            cache_limit: 48_819_791_680,
        };
        let grant = admit(measured, 63_050_041_152, Some(100_000_000_000)).unwrap();
        assert_eq!(grant.nonallocator_overhead, 116_310_640);
        assert_eq!(grant.physical_ceiling, 100_000_000_000);
        assert_eq!(grant.requested_cache_limit, 48_819_791_680);
        assert_eq!(grant.effective_cache_limit, 36_833_648_208);
        assert_eq!(grant.full_envelope, 100_000_000_000);
        assert!(grant.full_envelope <= grant.physical_ceiling);
    }
    #[test]
    fn numeric_refuses_when_only_active_not_full_physical_fits() {
        let measured = Host {
            cache_limit: 11_142_168_576,
            ..host()
        };
        let active = 69_080_366_523;
        assert!(admit_full(measured, active, Some(100_000_000_000)).is_ok());
        assert!(admit(measured, active, Some(75_000_000_000)).is_ok());
        assert!(admit_full(measured, active, Some(75_000_000_000)).is_err());
        assert!(admit_full(
            Host {
                available: 75_000_000_000,
                ..measured
            },
            active,
            None
        )
        .is_err());
        assert!(admit_full(
            Host {
                pressure: 2,
                ..measured
            },
            active,
            None
        )
        .is_err());
    }
    #[test]
    fn frozen_numeric_reserve_survives_zero_actual_allocator_cache() {
        let measured = Host {
            cache_limit: 0,
            ..host()
        };
        let active = 69_080_366_523;
        let reserve = 11_142_168_576;
        // The previous min(actual_cache, reserve) estimator wrongly admitted.
        assert!(admit_full(measured, active, Some(75_000_000_000)).is_ok());
        assert!(admit_numeric_full(measured, active, reserve, Some(75_000_000_000)).is_err());
        assert_eq!(
            admit_numeric_full(measured, active, reserve, Some(100_000_000_000)).unwrap(),
            80_222_535_099 + GIB / 2
        );
        assert_eq!(
            measured.cache_limit, 0,
            "reservation must not alter allocator policy"
        );
    }

    #[test]
    fn numeric_scoped_clamps_only_free_cache_and_types_the_grant() {
        let measured = Host {
            cache_limit: 0,
            ..host()
        };
        let active = 69_080_366_523;
        let reserve = 11_142_168_576;
        let (ceiling, grant) =
            admit_numeric_scoped(measured, active, reserve, Some(100_000_000_000)).unwrap();
        assert_eq!(ceiling, 80_222_535_099 + GIB / 2);
        assert_eq!(grant.physical_ceiling, ceiling);
        assert_eq!(grant.requested_cache_limit, reserve);
        assert_eq!(grant.effective_cache_limit, reserve);
        assert_eq!(grant.full_envelope, ceiling);

        let with_existing_cache = Host {
            cache_limit: reserve * 2,
            ..host()
        };
        let (ceiling, grant) =
            admit_numeric_scoped(with_existing_cache, active, reserve, Some(100_000_000_000))
                .unwrap();
        assert_eq!(grant.requested_cache_limit, reserve);
        assert_eq!(grant.effective_cache_limit, reserve);
        assert_eq!(grant.full_envelope, ceiling);
    }

    #[test]
    fn fresh_mac2_numeric_scope_preserves_active_budget_and_clamps_cache_exactly() {
        let measured = Host {
            total: 137_438_953_472,
            available: 98_274_099_200,
            recommended: 115_448_725_504,
            mlx_limit: 130_566_995_968,
            pressure: 1,
            baseline_physical: 25_526_824,
            baseline_active: 0,
            baseline_cache: 0,
            cache_limit: 48_819_791_680,
        };
        let active = 86_762_446_731;
        let frozen = 11_142_168_576;
        let (ceiling, grant) =
            admit_numeric_scoped(measured, active, frozen, Some(100_000_000_000)).unwrap();
        assert_eq!(ceiling, 93_385_921_064);
        assert_eq!(grant.active_envelope, active);
        assert_eq!(grant.nonallocator_overhead, 25_526_824);
        assert_eq!(grant.requested_cache_limit, frozen);
        assert_eq!(grant.effective_cache_limit, 6_597_947_509);
        assert_eq!(grant.full_envelope, ceiling);
        assert_eq!(frozen - grant.effective_cache_limit, 4_544_221_067);

        assert!(admit_numeric_scoped(
            measured,
            ceiling - grant.nonallocator_overhead + 1,
            frozen,
            Some(100_000_000_000)
        )
        .is_err());
        assert!(admit_numeric_scoped(measured, active, 0, Some(100_000_000_000)).is_err());
        assert!(admit_numeric_scoped(
            Host {
                pressure: 2,
                ..measured
            },
            active,
            frozen,
            Some(100_000_000_000)
        )
        .is_err());
        assert!(admit_numeric_scoped(measured, u64::MAX, frozen, None).is_err());
    }
    #[test]
    fn every_headroom_boundary_caps_cache_allowance_independently() {
        let roomy = Host {
            total: 128 * GIB,
            available: 128 * GIB,
            recommended: 128 * GIB,
            mlx_limit: 200 * GIB,
            ..host()
        };
        let edit = 100_983_708_416;
        assert_eq!(
            admit(roomy, edit, None).unwrap().physical_ceiling,
            (128 * GIB / 100) * 85
        );
        assert_eq!(
            admit(
                Host {
                    available: 105 * GIB,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap()
            .physical_ceiling,
            GIB + (105 * GIB / 100) * 95
        );
        assert_eq!(
            admit(
                Host {
                    recommended: 100 * GIB,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap()
            .physical_ceiling,
            100 * GIB + GIB / 2
        );
        assert_eq!(
            admit(
                Host {
                    mlx_limit: 120 * GIB,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap()
            .physical_ceiling,
            (120 * GIB / 100) * 85 + GIB / 2
        );
        assert_eq!(
            admit(roomy, edit, Some(102 * GIB))
                .unwrap()
                .physical_ceiling,
            102 * GIB
        );
        assert_eq!(
            admit(
                Host {
                    cache_limit: 0,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap()
            .physical_ceiling,
            edit + GIB / 2
        );
        assert!(admit(roomy, u64::MAX, None).is_err());
        assert!(admit(roomy, 0, None).is_err());
    }
    #[test]
    fn smaller_hosts_busy_hosts_pressure_and_unknown_census_fail_closed() {
        for host in [
            Host {
                total: 96 * GIB,
                ..host()
            },
            Host {
                available: 64 * GIB,
                ..host()
            },
            Host {
                recommended: 80 * GIB,
                ..host()
            },
            Host {
                mlx_limit: 80 * GIB,
                ..host()
            },
            Host {
                pressure: 2,
                ..host()
            },
            Host {
                pressure: 0,
                ..host()
            },
            Host { total: 0, ..host() },
            Host {
                available: 0,
                ..host()
            },
            Host {
                available: 129 * GIB,
                ..host()
            },
            Host {
                recommended: 0,
                ..host()
            },
            Host {
                mlx_limit: 0,
                ..host()
            },
        ] {
            assert!(admit(host, 100_983_708_416, None).is_err());
        }
    }
    #[test]
    fn actual_native_census_preserves_reserves_and_admits_original_edit() {
        // Run 37202513081, job 111436911459: normal pressure on a 128 GiB Mac.
        // vm_stat reports disjoint printed buckets at the actual 16384-byte page size.
        let raw = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 3449230.\nPages inactive: 2797979.\nPages speculative: 274298.\nPages purgeable: 25655.\nFile-backed pages: 2902110.\nPages occupied by compressor: 83848.\n";
        let available = reclaimable_bytes(raw).unwrap();
        assert_eq!(available, 106_848_370_688);
        let measured = Host {
            total: 137_438_953_472,
            available,
            recommended: 115_448_725_504,
            mlx_limit: 130_567_005_798,
            pressure: 1,
            baseline_physical: 108_855_944,
            baseline_active: 0,
            baseline_cache: 0,
            cache_limit: 86_753_458_944,
        };
        let envelope = 100_983_708_416;
        assert_eq!(
            admit(measured, envelope, None).unwrap().physical_ceiling,
            101_614_808_014
        );
        assert!(admit(
            Host {
                available: 102_354_272_256,
                ..measured
            },
            envelope,
            None
        )
        .is_err());
        assert!(admit(measured, envelope, Some(100_000_000_000)).is_err());
        assert!(admit(
            Host {
                pressure: 2,
                ..measured
            },
            envelope,
            None
        )
        .is_err());
        assert!(reclaimable_bytes(
            "free_count: 3723528\ninactive_count: 2797979\nspeculative_count: 274298\n"
        )
        .is_err());
    }
    #[test]
    fn printed_reclaimable_includes_speculative_once_without_overlapping_buckets() {
        let raw="Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages purgeable: 7.\nPages speculative: 3.\nPages occupied by compressor: 8.\n";
        assert_eq!(reclaimable_bytes(raw).unwrap(), 33 * 16384);
        assert_eq!(
            reclaimable_bytes(&raw.replace("16384", "4096")).unwrap(),
            33 * 4096
        );
        assert!(reclaimable_bytes(&raw.replace("Pages speculative: 3.", "")).is_err());
        assert!(
            reclaimable_bytes(&raw.replace("Pages speculative: 3.", "Pages speculative: -3."))
                .is_err()
        );
        assert!(reclaimable_bytes("Pages free: 100.").is_err());
    }

    #[test]
    fn exact_boundary_and_prior_zero_never_raise_the_caller_bound() {
        let required = 63_166_351_792;
        let boundary = Host {
            total: 137_438_953_472,
            available: 106_946_822_144,
            recommended: 115_448_725_504,
            mlx_limit: 130_567_005_798,
            pressure: 1,
            baseline_physical: 116_310_640,
            baseline_active: 0,
            baseline_cache: 0,
            cache_limit: 48_819_791_680,
        };
        let grant = admit(boundary, 63_050_041_152, Some(required)).unwrap();
        assert_eq!(grant.effective_cache_limit, 0);
        assert_eq!(grant.full_envelope, required);
        let zero = admit(
            Host {
                cache_limit: 0,
                ..boundary
            },
            63_050_041_152,
            Some(100_000_000_000),
        )
        .unwrap();
        assert_eq!(zero.effective_cache_limit, 0);
        assert_eq!(zero.full_envelope, required);
        let lower = admit(
            Host {
                cache_limit: 1_000_000_000,
                ..boundary
            },
            63_050_041_152,
            Some(100_000_000_000),
        )
        .unwrap();
        assert_eq!(lower.effective_cache_limit, 1_000_000_000);
        assert_eq!(lower.physical_ceiling, required + 1_000_000_000);
        assert_eq!(lower.full_envelope, lower.physical_ceiling);
    }

    #[test]
    fn checked_admission_rejects_every_relevant_overflow() {
        assert!(admit(
            Host {
                baseline_active: u64::MAX,
                baseline_cache: 1,
                ..host()
            },
            GIB,
            None
        )
        .is_err());
        assert!(admit(
            Host {
                baseline_physical: u64::MAX,
                available: 100,
                ..host()
            },
            GIB,
            None
        )
        .is_err());
        assert!(admit(
            Host {
                baseline_physical: GIB,
                baseline_active: 0,
                baseline_cache: 0,
                ..host()
            },
            u64::MAX,
            None
        )
        .is_err());
        assert!(admit(
            Host {
                cache_limit: u64::MAX,
                ..host()
            },
            GIB,
            None
        )
        .is_err());
    }

    #[derive(Clone)]
    struct FakeAllocator {
        limit: Rc<Cell<usize>>,
        events: Rc<RefCell<Vec<String>>>,
    }

    impl FakeAllocator {
        fn new(limit: usize) -> Self {
            Self {
                limit: Rc::new(Cell::new(limit)),
                events: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn mark(&self, event: &str) {
            self.events
                .borrow_mut()
                .push(format!("{event}:{}", self.limit.get()));
        }
    }

    impl CacheAllocator for FakeAllocator {
        fn set_cache_limit(&self, limit: usize) -> usize {
            let previous = self.limit.replace(limit);
            self.events
                .borrow_mut()
                .push(format!("set:{previous}->{limit}"));
            previous
        }

        fn clear_cache(&self) {
            self.events.borrow_mut().push("clear".into());
        }
    }

    #[derive(Clone)]
    struct PanicClearAllocator(FakeAllocator);

    impl CacheAllocator for PanicClearAllocator {
        fn set_cache_limit(&self, limit: usize) -> usize {
            self.0.set_cache_limit(limit)
        }

        fn clear_cache(&self) {
            panic!("clear failed");
        }
    }

    #[derive(Clone)]
    struct MismatchAllocator {
        inner: FakeAllocator,
        calls: Rc<Cell<usize>>,
    }

    impl CacheAllocator for MismatchAllocator {
        fn set_cache_limit(&self, limit: usize) -> usize {
            let observed = self.inner.set_cache_limit(limit);
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if call == 3 {
                observed.saturating_sub(1)
            } else {
                observed
            }
        }

        fn clear_cache(&self) {
            self.inner.clear_cache();
        }
    }

    fn scoped_test_grant(cache: u64) -> AdmissionGrant {
        AdmissionGrant {
            physical_ceiling: 1_000,
            nonallocator_overhead: 0,
            active_envelope: 100,
            requested_cache_limit: cache,
            effective_cache_limit: cache,
            full_envelope: 100 + cache,
        }
    }

    #[test]
    fn scoped_grant_is_nested_and_lives_across_load_and_train() {
        const PARENT_GRANT: usize = 36_833_648_208;
        const CHILD_REQUEST: usize = 48_819_791_680;
        let allocator = FakeAllocator::new(CHILD_REQUEST);
        {
            let outer =
                ScopedCacheGrant::enter(allocator.clone(), scoped_test_grant(PARENT_GRANT as u64))
                    .unwrap();
            assert_eq!(
                (outer.previous(), outer.effective()),
                (CHILD_REQUEST, PARENT_GRANT)
            );
            assert_eq!(outer.grant().effective_cache_limit, PARENT_GRANT as u64);
            allocator.mark("load");
            {
                let inner = ScopedCacheGrant::enter(
                    allocator.clone(),
                    scoped_test_grant(CHILD_REQUEST as u64),
                )
                .unwrap();
                assert_eq!(
                    (inner.previous(), inner.effective()),
                    (PARENT_GRANT, PARENT_GRANT)
                );
                allocator.mark("train");
            }
            allocator.mark("after_inner");
        }
        allocator.mark("after_outer");
        assert_eq!(
            allocator.events.borrow().as_slice(),
            [
                "set:48819791680->0",
                "set:0->36833648208",
                "set:36833648208->0",
                "set:0->36833648208",
                "clear",
                "load:36833648208",
                "set:36833648208->0",
                "set:0->36833648208",
                "set:36833648208->0",
                "set:0->36833648208",
                "clear",
                "train:36833648208",
                "set:36833648208->36833648208",
                "after_inner:36833648208",
                "set:36833648208->48819791680",
                "after_outer:48819791680",
            ]
        );
    }

    #[test]
    fn scoped_grant_restores_after_error_and_panic() {
        fn fail(allocator: FakeAllocator) -> Result<(), &'static str> {
            let _grant = ScopedCacheGrant::enter(allocator, scoped_test_grant(40)).unwrap();
            Err("training error")
        }

        let allocator = FakeAllocator::new(100);
        assert_eq!(fail(allocator.clone()), Err("training error"));
        assert_eq!(allocator.limit.get(), 100);

        let unwind_allocator = allocator.clone();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _grant = ScopedCacheGrant::enter(unwind_allocator, scoped_test_grant(30)).unwrap();
            panic!("training panic");
        }));
        assert!(panic.is_err());
        assert_eq!(allocator.limit.get(), 100);

        let clear_allocator = PanicClearAllocator(allocator.clone());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            ScopedCacheGrant::enter(clear_allocator, scoped_test_grant(20)).unwrap();
        }));
        assert!(panic.is_err());
        assert_eq!(allocator.limit.get(), 100);

        let mismatch = MismatchAllocator {
            inner: allocator.clone(),
            calls: Rc::new(Cell::new(0)),
        };
        assert!(ScopedCacheGrant::enter(mismatch, scoped_test_grant(25)).is_err());
        assert_eq!(allocator.limit.get(), 100);
    }

    #[test]
    fn diagnostic_cleanup_order_holds_for_success_error_and_panic() {
        #[derive(Clone, Copy)]
        enum Exit {
            Success,
            Error,
            Panic,
        }
        struct Marker {
            name: &'static str,
            events: Rc<RefCell<Vec<&'static str>>>,
        }
        impl Drop for Marker {
            fn drop(&mut self) {
                self.events.borrow_mut().push(self.name);
            }
        }
        fn exercise(exit: Exit, events: Rc<RefCell<Vec<&'static str>>>) -> Result<(), ()> {
            // Mirrors the diagnostic's declaration order. Native locals are
            // created last, so retirement runs while all safety guards live.
            let _watchdog = Marker {
                name: "watchdog",
                events: events.clone(),
            };
            let _cache_grant = Marker {
                name: "cache-grant",
                events: events.clone(),
            };
            let _bounds = Marker {
                name: "bounds",
                events: events.clone(),
            };
            let _retirement = Marker {
                name: "retirement",
                events: events.clone(),
            };
            let _native = Marker {
                name: "native",
                events,
            };
            match exit {
                Exit::Success => Ok(()),
                Exit::Error => Err(()),
                Exit::Panic => panic!("lifecycle test panic"),
            }
        }

        for exit in [Exit::Success, Exit::Error, Exit::Panic] {
            let events = Rc::new(RefCell::new(Vec::new()));
            let observed = events.clone();
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = exercise(exit, events);
            }));
            assert_eq!(
                observed.borrow().as_slice(),
                ["native", "retirement", "bounds", "cache-grant", "watchdog"]
            );
        }

        let source = include_str!("../../src/conditioning_velocity_diagnostic.rs");
        let watchdog = source
            .find("let guard = evidence::Footprint::start")
            .unwrap();
        let cache_grant = source.find("let _current_cache_grant =").unwrap();
        let bounds = source.find("let _bounds =").unwrap();
        let retirement = source.find("let _retire_on_drop = RetireOnDrop").unwrap();
        let native_locals = source.find("let mut cache = math::Cache::default").unwrap();
        assert!(watchdog < cache_grant);
        assert!(cache_grant < bounds);
        assert!(bounds < retirement);
        assert!(retirement < native_locals);
    }
}
