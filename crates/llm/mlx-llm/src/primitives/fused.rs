//! Fused-versus-reference route tally (epic sc-24432 E3).
//!
//! Moved to [`mlx_layers::prism::fused`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use mlx_layers::prism::fused::{
    fused_tally, note_fused, note_reference, FusedTally, REASON_CPU_STREAM, REASON_DISABLED,
    REASON_FORCED_REFERENCE, REASON_SHAPE,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tally_counts_each_route_on_this_thread() {
        let start = fused_tally();
        note_fused();
        note_reference(REASON_CPU_STREAM);
        let since = fused_tally().since(&start);
        assert_eq!((since.fused, since.reference), (1, 1));
        assert_eq!(since.label(), "mixed");
        assert_eq!(since.reference_reason, Some(REASON_CPU_STREAM));
        // Another thread's runs never reach this one's tally.
        std::thread::spawn(note_fused).join().unwrap();
        assert_eq!(fused_tally().since(&start).fused, 1);
    }
}
