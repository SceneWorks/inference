//! The iOS anchor law lives in [`mlx_gen::ios_anchor_law`] (shared with the providers' CPU
//! additivity proofs); this module binds it to the evidence table.

use crate::evidence::AnchorRow;
pub use mlx_gen::ios_anchor_law::*;

/// A completed anchor row's measurement (largest growth across repeats, per phase); `None` for a
/// refused anchor.
pub fn measured_anchor(row: &AnchorRow) -> Option<MeasuredAnchor> {
    Some(MeasuredAnchor {
        demand: row.demand_bytes()?,
        phases: row.phase_demands(),
        area: area(row.key.width, row.key.height, row.key.batch),
    })
}
