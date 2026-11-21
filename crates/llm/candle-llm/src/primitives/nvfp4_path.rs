//! NVFP4 projection forward dispatch and telemetry (epic sc-24128, story sc-24136).
//!
//! Moved to [`candle_layers::nvfp4_path`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use candle_layers::nvfp4_path::{
    forward, note_cublaslt, note_gemv, nvfp4_gemv_enabled, nvfp4_gemv_policy_guard,
    nvfp4_path_tally, set_nvfp4_gemv, Nvfp4GemvPolicyGuard, Nvfp4PathTally, NVFP4_GEMV_ENV,
    NVFP4_GEMV_MAX_ROWS, REASON_DISABLED,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_deltas_and_labels() {
        let start = nvfp4_path_tally();
        assert_eq!(nvfp4_path_tally().since(&start).label(), "none");
        note_gemv();
        assert_eq!(nvfp4_path_tally().since(&start).label(), "gemv");
        note_cublaslt("rows");
        let d = nvfp4_path_tally().since(&start);
        assert_eq!((d.gemv, d.cublaslt), (1, 1));
        assert_eq!(d.cublaslt_reason, Some("rows"));
        assert_eq!(d.label(), "mixed");
        let later = nvfp4_path_tally();
        note_cublaslt("dtype");
        let d = nvfp4_path_tally().since(&later);
        assert_eq!(
            d,
            Nvfp4PathTally {
                gemv: 0,
                cublaslt: 1,
                cublaslt_reason: Some("dtype")
            }
        );
        assert_eq!(d.label(), "cublaslt");
        let now = nvfp4_path_tally();
        note_gemv();
        assert_eq!(nvfp4_path_tally().since(&now).cublaslt_reason, None);
    }

    #[test]
    fn tally_is_per_thread() {
        let before = nvfp4_path_tally();
        note_gemv();
        let other = std::thread::spawn(|| {
            let start = nvfp4_path_tally();
            note_cublaslt("x");
            nvfp4_path_tally().since(&start)
        })
        .join()
        .unwrap();
        assert_eq!(other.cublaslt, 1);
        assert_eq!(nvfp4_path_tally().since(&before).cublaslt, 0);
    }

    #[test]
    fn runtime_override_beats_the_environment_and_can_be_cleared() {
        let _policy = nvfp4_gemv_policy_guard(None);
        let from_env = nvfp4_gemv_enabled();
        set_nvfp4_gemv(Some(false));
        assert!(!nvfp4_gemv_enabled());
        set_nvfp4_gemv(Some(true));
        assert!(nvfp4_gemv_enabled());
        set_nvfp4_gemv(None);
        assert_eq!(nvfp4_gemv_enabled(), from_env);
    }

    #[test]
    fn the_policy_guard_restores_the_switch_it_found() {
        let before = nvfp4_gemv_policy_guard(None).found();
        {
            let _held = nvfp4_gemv_policy_guard(Some(false));
            assert!(!nvfp4_gemv_enabled());
            set_nvfp4_gemv(Some(true));
            assert!(nvfp4_gemv_enabled());
        }
        let after = nvfp4_gemv_policy_guard(None);
        assert_eq!(
            after.found(),
            before,
            "dropping the guard restores the switch"
        );
    }
}
