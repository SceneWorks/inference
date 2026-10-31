//! Shared I/O progress contract. See `load_progress` for request/lazy-thread propagation.
pub use load_progress::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    #[test]
    fn seal_memo_hit_only_checks_and_cold_read_reports_exact_hash_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.bin");
        std::fs::write(&file, vec![42; 8193]).unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let out = events.clone();
        let _observer = LoadActivityObserver::new(move |v| out.lock().unwrap().push(v));
        let first = crate::ArtifactSeal::pin(&file).unwrap();
        let reports = events.lock().unwrap().clone();
        assert!(reports
            .iter()
            .flatten()
            .any(|v| v.phase == LoadActivityPhase::Verifying
                && v.bytes_read == 8193
                && v.total_bytes == Some(8193)));
        assert_eq!(reports.last(), Some(&None));
        events.lock().unwrap().clear();
        let warm = crate::ArtifactSeal::pin(&file).unwrap();
        assert_eq!(warm.content_sha256(), first.content_sha256());
        let reports = events.lock().unwrap();
        assert!(reports
            .iter()
            .flatten()
            .all(|v| v.phase == LoadActivityPhase::Checking && v.bytes_read == 0));
        assert!(reports
            .iter()
            .flatten()
            .any(|v| v.files_checked == 1 && v.total_files == Some(1)));
        assert_eq!(reports.last(), Some(&None));
    }
}
