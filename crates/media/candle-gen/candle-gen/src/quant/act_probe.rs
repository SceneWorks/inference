//! Activation distribution probe, re-exported from [`candle_layers::act_probe`] (`SW-6955`).

pub use candle_layers::act_probe::*;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_layers::quant::{ActPrecision, OutlierClass, OutlierSparsity};

    #[test]
    fn summarize_reports_worst_case_and_partition_verdict() {
        let mk = |layer: &str, step: usize, act, benign: f64| ActRecord {
            layer: layer.to_string(),
            step,
            act,
            sparsity: OutlierSparsity {
                total_blocks: 1000,
                outlier_blocks: ((1.0 - benign) * 1000.0).round() as usize,
                benign_fraction: benign,
                robust_scale: 1.0,
                max_crush_ratio: 10.0,
                tau: 20.0,
            },
        };
        let recs = vec![
            mk("a", 0, ActPrecision::W4A4, 0.999),
            mk("a", 1, ActPrecision::W4A4, 0.996), // worst step still benign
            mk("b", 0, ActPrecision::W4A4, 0.999),
            mk("b", 1, ActPrecision::W4A4, 0.5), // collapses at step 1 → partition broken
        ];
        let s = summarize(&recs);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].layer, "a");
        assert_eq!(s[0].steps, 2);
        assert!((s[0].min_benign_fraction - 0.996).abs() < 1e-9);
        assert_eq!(s[0].worst_class, OutlierClass::Benign);
        assert!(s[0].partition_holds());
        // `b` is assigned W4A4 but measures Dense at its worst step — the gate must catch it.
        assert_eq!(s[1].worst_class, OutlierClass::Dense);
        assert!(!s[1].partition_holds());
    }
}
