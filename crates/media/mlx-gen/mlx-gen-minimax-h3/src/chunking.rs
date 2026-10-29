//! Temporal chunk bookkeeping for the video VAE decode (`klvae.py::decode_temporal`).
//!
//! The pure-integer plan is backend-neutral and lives in
//! [`gen_core_h3::h3_chunking`] (sc-27256), shared with the Candle twin so the two backends
//! cannot disagree about the frame plan; this module re-exports it so `chunking::*` paths stay
//! stable, and binds this crate's [`MiniMaxH3VaeConfig`] to it.

pub use gen_core_h3::h3_chunking::*;

use crate::config::MiniMaxH3VaeConfig;

impl H3TemporalConfig for MiniMaxH3VaeConfig {
    fn clip_length(&self) -> i32 {
        self.clip_length
    }

    fn token_drop(&self) -> i32 {
        self.token_drop
    }

    fn patch_size_t(&self) -> i128 {
        i128::from(self.patch_size_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_from_config_matches_the_raw_knobs() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(
            TemporalGeometry::new(&cfg).unwrap(),
            TemporalGeometry::from_parts(17, 3, 4).unwrap()
        );
        assert!(TemporalGeometry::isolated_frames_unsupported());
    }
}
