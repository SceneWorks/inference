//! Independent callable-route reconciliation (`SW-6982`, `PIPE-18`, `B1`, `E4`).
//!
//! Reconciles candidate `sga.capabilities/2` exports independently of provider registration code,
//! verifying that every required shipping callable route cell (`{platform}/{backend}/{route_id}/operation/{mode}`)
//! is present and advertises the expected operation mode, and that `XL-4` `TextLlmRegistry`
//! StarVector IDs remain intact.

use std::fmt;

use gen_core::export::CapabilitiesExportV2;

/// One required callable route + operation cell in a candidate capability export.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequiredRouteCell {
    pub platform: &'static str,
    pub backend: &'static str,
    pub route_id: &'static str,
    pub operation: &'static str,
}

impl RequiredRouteCell {
    pub const fn new(
        platform: &'static str,
        backend: &'static str,
        route_id: &'static str,
        operation: &'static str,
    ) -> Self {
        Self {
            platform,
            backend,
            route_id,
            operation,
        }
    }

    /// Canonical cell identifier: `{platform}/{backend}/{route_id}/operation/{operation}`.
    pub fn cell_id(&self) -> String {
        format!(
            "{}/{}/{}/operation/{}",
            self.platform, self.backend, self.route_id, self.operation
        )
    }
}

/// One required `TextLlmRegistry` entry for `XL-4` backward compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequiredLlmCompatibilityEntry {
    pub platform: &'static str,
    pub backend: &'static str,
    pub llm_id: &'static str,
}

/// All 4 StarVector vector-route cells required across `macos/mlx` and `cuda/candle` (`PIPE-18`, `B1`).
pub const STARVECTOR_REQUIRED_CELLS: &[RequiredRouteCell] = &[
    RequiredRouteCell::new("macos", "mlx", "starvector_1b", "image_to_svg"),
    RequiredRouteCell::new("macos", "mlx", "starvector_8b", "image_to_svg"),
    RequiredRouteCell::new("cuda", "candle", "starvector_1b", "image_to_svg"),
    RequiredRouteCell::new("cuda", "candle", "starvector_8b", "image_to_svg"),
];

/// All 4 Candle Qwen edit alias cells + Qwen control (`B1`, `E4`).
pub const CANDLE_QWEN_EDIT_REQUIRED_CELLS: &[RequiredRouteCell] = &[
    RequiredRouteCell::new("cuda", "candle", "qwen_image_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "qwen_image_edit_2509", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "qwen_image_edit_2511", "image_to_image"),
    RequiredRouteCell::new(
        "cuda",
        "candle",
        "qwen_image_edit_2511_lightning",
        "image_to_image",
    ),
    RequiredRouteCell::new("cuda", "candle", "qwen_image_control", "control"),
];

/// All shipping bespoke provider cells that must be registered callable routes (`REG-6`, `PIPE-18`).
pub const SHIPPING_BESPOKE_REQUIRED_CELLS: &[RequiredRouteCell] = &[
    // Candle Kolors IP & Control
    RequiredRouteCell::new("cuda", "candle", "kolors_ip_adapter", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "kolors_control", "control"),
    // Candle FLUX.1 IP & Control
    RequiredRouteCell::new("cuda", "candle", "flux1_dev_ip_adapter", "image_to_image"),
    RequiredRouteCell::new(
        "cuda",
        "candle",
        "flux1_schnell_ip_adapter",
        "image_to_image",
    ),
    RequiredRouteCell::new("cuda", "candle", "flux1_dev_control", "control"),
    // Candle FLUX.2 Edit & Control
    RequiredRouteCell::new("cuda", "candle", "flux2_klein_9b_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "flux2_dev_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "flux2_dev_control", "control"),
    // Candle SDXL Edit, Detail, Control, IP-Adapter
    RequiredRouteCell::new("cuda", "candle", "sdxl_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "sdxl_detail", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "sdxl_control", "control"),
    RequiredRouteCell::new("cuda", "candle", "sdxl_ip_adapter", "image_to_image"),
    // Candle Z-Image Control
    RequiredRouteCell::new("cuda", "candle", "z_image_control", "control"),
    RequiredRouteCell::new("cuda", "candle", "z_image_turbo_control", "control"),
    // Candle & MLX PuLID + InstantID
    RequiredRouteCell::new("cuda", "candle", "pulid_flux", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "instantid", "image_to_image"),
    RequiredRouteCell::new("macos", "mlx", "pulid_flux", "image_to_image"),
    RequiredRouteCell::new("macos", "mlx", "instantid", "image_to_image"),
    // Candle Krea Edit & Control
    RequiredRouteCell::new("cuda", "candle", "krea_2_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "krea_2_turbo_edit", "image_to_image"),
    RequiredRouteCell::new("cuda", "candle", "krea_2_turbo_control", "control"),
];

/// `XL-4` compatibility entries ensuring `TextLlmRegistry` StarVector IDs stay intact.
pub const STARVECTOR_LLM_COMPATIBILITY_ENTRIES: &[RequiredLlmCompatibilityEntry] = &[
    RequiredLlmCompatibilityEntry {
        platform: "macos",
        backend: "mlx",
        llm_id: "mlx-starvector-1b",
    },
    RequiredLlmCompatibilityEntry {
        platform: "macos",
        backend: "mlx",
        llm_id: "mlx-starvector-8b",
    },
    RequiredLlmCompatibilityEntry {
        platform: "cuda",
        backend: "candle",
        llm_id: "candle-starvector-1b",
    },
    RequiredLlmCompatibilityEntry {
        platform: "cuda",
        backend: "candle",
        llm_id: "candle-starvector-8b",
    },
];

/// One missing or mismatched callable route cell found during reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconciliationFailure {
    pub cell: String,
    pub detail: String,
}

/// Aggregated error returned when candidate capability exports fail reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconciliationError {
    pub failures: Vec<ReconciliationFailure>,
}

impl ReconciliationError {
    /// Returns every failing cell identifier in deterministic order.
    pub fn missing_cells(&self) -> Vec<&str> {
        self.failures.iter().map(|f| f.cell.as_str()).collect()
    }
}

impl fmt::Display for ReconciliationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "callable-route reconciliation failed with {} violation(s):",
            self.failures.len()
        )?;
        for failure in &self.failures {
            writeln!(f, "  - {}: {}", failure.cell, failure.detail)?;
        }
        Ok(())
    }
}

impl std::error::Error for ReconciliationError {}

/// Reconcile a slice of candidate `sga.capabilities/2` exports against an arbitrary set of required
/// route cells and optional `XL-4` LLM compatibility entries.
///
/// Only checks cells whose `(platform, backend)` matches at least one supplied export, unless
/// `require_all_platforms` is true (in which case an absent `(platform, backend)` export itself
/// fails every cell targeting it).
pub fn reconcile_exports(
    exports: &[&CapabilitiesExportV2],
    required_cells: &[RequiredRouteCell],
    required_llms: &[RequiredLlmCompatibilityEntry],
    require_all_platforms: bool,
) -> std::result::Result<(), ReconciliationError> {
    let mut failures = Vec::new();

    for cell in required_cells {
        let export = exports
            .iter()
            .copied()
            .find(|e| e.platform == cell.platform && e.backend == cell.backend);
        let Some(export) = export else {
            if require_all_platforms {
                failures.push(ReconciliationFailure {
                    cell: cell.cell_id(),
                    detail: format!(
                        "no candidate export supplied for platform={} backend={}",
                        cell.platform, cell.backend
                    ),
                });
            }
            continue;
        };

        match export.routes.get(cell.route_id) {
            None => failures.push(ReconciliationFailure {
                cell: cell.cell_id(),
                detail: format!(
                    "route `{}` is missing from {} ({}) candidate export",
                    cell.route_id, export.platform, export.backend
                ),
            }),
            Some(route) => {
                if !route.modes.iter().any(|m| m == cell.operation) {
                    failures.push(ReconciliationFailure {
                        cell: cell.cell_id(),
                        detail: format!(
                            "route `{}` advertises modes {:?}, missing required operation `{}`",
                            cell.route_id, route.modes, cell.operation
                        ),
                    });
                }
            }
        }
    }

    for entry in required_llms {
        let export = exports
            .iter()
            .copied()
            .find(|e| e.platform == entry.platform && e.backend == entry.backend);
        let Some(export) = export else {
            if require_all_platforms {
                failures.push(ReconciliationFailure {
                    cell: format!("{}/{}/llm/{}", entry.platform, entry.backend, entry.llm_id),
                    detail: format!(
                        "no candidate export supplied for platform={} backend={}",
                        entry.platform, entry.backend
                    ),
                });
            }
            continue;
        };
        if !export.llm_models.contains_key(entry.llm_id) {
            failures.push(ReconciliationFailure {
                cell: format!("{}/{}/llm/{}", entry.platform, entry.backend, entry.llm_id),
                detail: format!(
                    "XL-4 compatibility LLM provider `{}` is missing from {} ({}) export",
                    entry.llm_id, export.platform, export.backend
                ),
            });
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(ReconciliationError { failures })
    }
}

/// Reconcile candidate exports against the full `SW-6982` (`PIPE-18` / `B1` / `E4`) contract:
/// StarVector 1B/8B vector cells, all 4 Candle Qwen edit aliases + control, every shipping bespoke
/// route, and `XL-4` StarVector `TextLlmRegistry` entries.
pub fn reconcile_callable_routes(
    exports: &[&CapabilitiesExportV2],
    require_all_platforms: bool,
) -> std::result::Result<(), ReconciliationError> {
    let mut cells = Vec::with_capacity(
        STARVECTOR_REQUIRED_CELLS.len()
            + CANDLE_QWEN_EDIT_REQUIRED_CELLS.len()
            + SHIPPING_BESPOKE_REQUIRED_CELLS.len(),
    );
    cells.extend_from_slice(STARVECTOR_REQUIRED_CELLS);
    cells.extend_from_slice(CANDLE_QWEN_EDIT_REQUIRED_CELLS);
    cells.extend_from_slice(SHIPPING_BESPOKE_REQUIRED_CELLS);
    reconcile_exports(
        exports,
        &cells,
        STARVECTOR_LLM_COMPATIBILITY_ENTRIES,
        require_all_platforms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::starvector::starvector_8b;
    use gen_core::export::{
        ExportedBespokeSurfaceV2, ExportedEmbeddersV2, ExportedLlmEntry, ExportedRouteV2,
        ExportedTransformsV2, CAPABILITIES_EXPORT_SCHEMA_VERSION,
    };
    use std::collections::BTreeMap;

    fn dummy_route(id: &'static str, backend: &'static str, mode: &str) -> ExportedRouteV2 {
        let mut route = crate::starvector::route_candle_1b(crate::starvector::unavailable_loader);
        route.route_id = id;
        route.backend = backend;
        route.mac_only = backend == "mlx";
        let mut exported = ExportedRouteV2::from_route_record(&route, None, false);
        exported.modes = vec![mode.to_string()];
        exported
    }

    fn dummy_export(platform: &'static str, backend: &'static str) -> CapabilitiesExportV2 {
        let mut routes = BTreeMap::new();
        for cell in STARVECTOR_REQUIRED_CELLS
            .iter()
            .chain(CANDLE_QWEN_EDIT_REQUIRED_CELLS.iter())
            .chain(SHIPPING_BESPOKE_REQUIRED_CELLS.iter())
        {
            if cell.platform == platform && cell.backend == backend {
                routes.insert(
                    cell.route_id.to_string(),
                    dummy_route(cell.route_id, backend, cell.operation),
                );
            }
        }
        let mut llm_models = BTreeMap::new();
        for entry in STARVECTOR_LLM_COMPATIBILITY_ENTRIES {
            if entry.platform == platform && entry.backend == backend {
                let desc = starvector_8b::text_descriptor(entry.llm_id, backend);
                llm_models.insert(
                    entry.llm_id.to_string(),
                    ExportedLlmEntry::from_descriptor(&desc),
                );
            }
        }
        CapabilitiesExportV2 {
            schema_version: CAPABILITIES_EXPORT_SCHEMA_VERSION.to_string(),
            backend: backend.to_string(),
            platform: platform.to_string(),
            feature_set: format!("{platform}-{backend}"),
            audio_backend: None,
            advanced_quant_tiers: Vec::new(),
            rejected_quants: BTreeMap::new(),
            snapshot_preparer_backends: Vec::new(),
            audio_snapshot_preparer_backends: Vec::new(),
            routes,
            audio_routes: BTreeMap::new(),
            feature_routes: BTreeMap::new(),
            llm_models,
            trainers: BTreeMap::new(),
            captioners: BTreeMap::new(),
            transcribers: BTreeMap::new(),
            embedders: ExportedEmbeddersV2::default(),
            transforms: ExportedTransformsV2::default(),
            imported_models: Vec::new(),
            checkpoint_adapters: BTreeMap::new(),
            bespoke_waivers: ExportedBespokeSurfaceV2::default(),
        }
    }

    #[test]
    fn negative_control_deleting_starvector_bindings_fails_naming_all_four_cells() {
        let mut macos = dummy_export("macos", "mlx");
        let mut cuda = dummy_export("cuda", "candle");
        reconcile_callable_routes(&[&macos, &cuda], true).expect("complete exports reconcile");

        macos.routes.remove("starvector_1b");
        macos.routes.remove("starvector_8b");
        cuda.routes.remove("starvector_1b");
        cuda.routes.remove("starvector_8b");

        let err = reconcile_callable_routes(&[&macos, &cuda], true)
            .expect_err("deleting StarVector bindings must fail reconciliation");
        let missing = err.missing_cells();
        assert_eq!(
            missing,
            vec![
                "macos/mlx/starvector_1b/operation/image_to_svg",
                "macos/mlx/starvector_8b/operation/image_to_svg",
                "cuda/candle/starvector_1b/operation/image_to_svg",
                "cuda/candle/starvector_8b/operation/image_to_svg",
            ]
        );
    }

    #[test]
    fn negative_control_removing_any_shipping_bespoke_route_fails_naming_that_cell() {
        for cell in SHIPPING_BESPOKE_REQUIRED_CELLS
            .iter()
            .chain(CANDLE_QWEN_EDIT_REQUIRED_CELLS.iter())
        {
            let mut macos = dummy_export("macos", "mlx");
            let mut cuda = dummy_export("cuda", "candle");
            if cell.platform == "macos" {
                macos.routes.remove(cell.route_id);
            } else {
                cuda.routes.remove(cell.route_id);
            }
            let err = reconcile_callable_routes(&[&macos, &cuda], true).expect_err(&format!(
                "removing shipping route `{}` must fail reconciliation",
                cell.route_id
            ));
            assert_eq!(err.missing_cells(), vec![cell.cell_id().as_str()]);
        }
    }
}
