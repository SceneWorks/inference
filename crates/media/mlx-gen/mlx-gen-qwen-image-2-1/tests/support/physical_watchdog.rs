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

/// Reserve 15% of physical RAM and 5% of currently reclaimable pages, and keep
/// the engine's 15% MLX-limit headroom. The recommended Metal working set also bounds allocator residency;
/// measured non-allocator baseline bytes are allowed separately. The sampler
/// still aborts if cache retention consumes this budget before a step finishes.
pub fn admit(host: Host, envelope: u64, explicit_cap: Option<u64>) -> Result<u64, String> {
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
    let overhead = host
        .baseline_physical
        .saturating_sub(host.baseline_active.saturating_add(host.baseline_cache));
    let fraction = |bytes: u64, numerator: u64| (bytes / 100) * numerator;
    let safe_cap = fraction(host.total, 85)
        .min(
            host.baseline_physical
                .saturating_add(fraction(host.available, 95)),
        )
        .min(host.recommended.saturating_add(overhead))
        .min(fraction(host.mlx_limit, 85).saturating_add(overhead))
        .min(explicit_cap.unwrap_or(u64::MAX));
    let required = envelope.saturating_add(overhead);
    if required > safe_cap {
        return Err(format!("selected preflight {envelope} + measured overhead {overhead} requires {required} bytes; verified physical cap is {safe_cap} bytes"));
    }
    Ok(required.saturating_add(host.cache_limit).min(safe_cap))
}

/// The test-only materialized diagnostic reserves its full live plus free-cache
/// envelope before constructing any tensor. Ordinary training admission is unchanged.
pub fn admit_full(host: Host, envelope: u64, explicit_cap: Option<u64>) -> Result<u64, String> {
    let ceiling = admit(host, envelope, explicit_cap)?;
    let overhead = host
        .baseline_physical
        .saturating_sub(host.baseline_active.saturating_add(host.baseline_cache));
    let required = envelope
        .saturating_add(overhead)
        .saturating_add(host.cache_limit);
    if required > ceiling {
        return Err(format!("numeric full physical envelope {required} exceeds verified cap {ceiling}; active-only admission is insufficient"));
    }
    Ok(ceiling)
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
        let cap = admit(host(), envelope, None).unwrap();
        assert!(cap >= envelope + GIB / 2);
        assert_eq!(cap, 104 * GIB + GIB / 2);
        assert!(cap < 128 * GIB * 85 / 100);
        assert!(admit(host(), envelope, Some(100_000_000_000)).is_err());
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
    fn every_headroom_boundary_caps_cache_allowance_independently() {
        let roomy = Host {
            total: 128 * GIB,
            available: 128 * GIB,
            recommended: 128 * GIB,
            mlx_limit: 200 * GIB,
            ..host()
        };
        let edit = 100_983_708_416;
        assert_eq!(admit(roomy, edit, None).unwrap(), (128 * GIB / 100) * 85);
        assert_eq!(
            admit(
                Host {
                    available: 105 * GIB,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap(),
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
            .unwrap(),
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
            .unwrap(),
            (120 * GIB / 100) * 85 + GIB / 2
        );
        assert_eq!(admit(roomy, edit, Some(102 * GIB)).unwrap(), 102 * GIB);
        assert_eq!(
            admit(
                Host {
                    cache_limit: 0,
                    ..roomy
                },
                edit,
                None
            )
            .unwrap(),
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
                recommended: 0,
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
        assert_eq!(admit(measured, envelope, None).unwrap(), 101_614_808_014);
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
}
