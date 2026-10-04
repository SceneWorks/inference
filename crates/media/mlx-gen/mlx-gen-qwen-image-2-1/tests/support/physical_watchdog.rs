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

/// vm_stat's free and inactive pages are reclaimable; purgeable pages may also
/// be inactive and must not be added a second time. Speculative/compressed and
/// wired pages are recorded by the harness but excluded from admission.
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
    fn reclaimable_pages_do_not_double_count_purgeable_or_speculative() {
        let raw="Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\nPages purgeable: 7.\nPages speculative: 3.\nPages occupied by compressor: 8.\n";
        assert_eq!(reclaimable_bytes(raw).unwrap(), 30 * 16384);
        assert!(reclaimable_bytes("Pages free: 100.").is_err());
    }
}
