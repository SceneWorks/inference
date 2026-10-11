//! Opt-in MiniMax-H3 learned sparse attention. An absent or disabled request preserves dense execution.

/// Request controls use the frozen Veda defaults. Providers report what actually executed.
#[derive(Clone, Debug, PartialEq)]
pub struct VedaRequest {
    pub enabled: bool,
    /// Percentage of generated-video tile pairs skipped; frozen default is 90.
    pub sparsity_percent: f64,
}
impl Default for VedaRequest {
    fn default() -> Self {
        Self {
            enabled: false,
            sparsity_percent: 90.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VedaExecutionReport {
    pub requested: bool,
    pub kernel: Option<String>,
    pub predictor: Option<String>,
    pub plan: Option<String>,
    pub exact_plan: bool,
    pub predictor_load_attempts: u64,
    pub predictor_score_calls: u64,
    pub sparse_calls: u64,
    pub dense_calls: u64,
    pub selected_token_pairs: u64,
    pub dense_token_pairs: u64,
    /// Maximum accounted workspace of any executed attention call, excluding the caller's input QKV.
    pub workspace_bytes: u64,
    /// Engine-observed reason for dense fallback; never inferred from requested settings.
    pub fallback_reason: Option<String>,
}
