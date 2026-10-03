//! Cross-backend conformance fixture for the compressed-KV policy (epic sc-20669, story sc-20683).
//!
//! One table of requests — policy × table family × loaded decoder (plain, outside the fused
//! reader's geometry, hybrid recurrent) × prompt × token budget × batch × request shape — and the
//! decision every backend must reach for each. A backend runs the table through its own production
//! planner (MLX's provider plan, Candle's provider plan) with [`kv_policy_conformance`]; because
//! every backend checks against the same table, MLX and Candle reach the same decision and reason
//! for every request, except where backend availability differs: a request a backend with a fused
//! compressed-domain reader runs compressed, a backend without one ([`KvReader::Unavailable`], Candle
//! on every device) runs dense as [`KvCacheFallbackReason::ReaderUnavailable`] — never claiming a
//! compressed format or compressed counters.
//!
//! The table's context points are derived from the shape of
//! [`KV_COMPRESSION_QUALIFICATIONS`] (each row's bounds and their neighbours), so a requalified row
//! moves the fixture with it; the expected decision is this module's own restatement of the
//! policy's fixed order (policy, batch, family, prompt minimum, final-context maximum, request
//! shape, geometry), independent of
//! [`core_llm::qualify_kv_compression`]'s implementation.

use core_llm::{
    KvCacheCounters, KvCacheFallbackReason, KvCacheReport, KvCompressionFormat,
    KvCompressionPolicy, KvModelFamily, KV_CACHE_FORMAT_VERSION, KV_COMPRESSION_QUALIFICATIONS,
};

/// Whether the backend under test has a fused compressed-domain KV reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvReader {
    /// A fused reader serves qualified requests (MLX on Metal).
    Fused,
    /// No fused reader on this backend (Candle, CPU or CUDA): every request runs dense.
    Unavailable,
}

/// The decision for one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvPolicyDecision {
    /// Run compressed in this format.
    Compressed(KvCompressionFormat),
    /// Run dense for this reason.
    Dense(KvCacheFallbackReason),
}

/// What a backend's planner decided for one case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvBackendDecision {
    /// The backend selected the compressed cache in this format.
    Compressed(KvCompressionFormat),
    /// The backend runs dense and reports this.
    Dense(KvCacheReport),
}

/// The loaded decoder a case plans on. A backend runs the geometry and hybrid cases on decoders it
/// actually loads, so its own geometry and decoder-shape extraction is under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCaseDecoder {
    /// A plain causal decoder the fused reader can read (head dimension 64, 128 or 256).
    Plain,
    /// A plain causal decoder outside the fused reader's geometry (an unsupported head
    /// dimension): [`KvCacheFallbackReason::UnsupportedGeometry`] on every backend.
    UnsupportedGeometry,
    /// A hybrid recurrent decoder (Qwen3.5/3.6): no compressed cache, so
    /// [`KvCacheFallbackReason::UnsupportedRequest`].
    Hybrid,
}

impl KvCaseDecoder {
    /// Every decoder of the table.
    pub const ALL: [Self; 3] = [Self::Plain, Self::UnsupportedGeometry, Self::Hybrid];
}

/// One request of the fixture table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvPolicyCase {
    /// The request's opt-in.
    pub policy: KvCompressionPolicy,
    /// The loaded model's table family; `None` is a model the table cannot name (a backend loads a
    /// Llama-dispatched non-llama checkpoint such as Mistral for it).
    pub family: Option<KvModelFamily>,
    /// The loaded decoder (armed as `family` by the backend's test seam).
    pub decoder: KvCaseDecoder,
    /// Tokens prefilled before decoding starts.
    pub context_tokens: u64,
    /// The request's token budget (`max_new_tokens`): the final context is the prompt plus this.
    pub max_new_tokens: u32,
    /// Sequences decoding together.
    pub batch: u64,
    /// Whether the request carries multimodal content (a path the compressed cache is not wired
    /// through).
    pub multimodal: bool,
    /// The decision a backend with a fused reader reaches.
    pub expected: KvPolicyDecision,
}

impl KvPolicyCase {
    /// The decision a backend with `reader` must reach: [`Self::expected`], except that a backend
    /// without a fused reader runs a would-be-compressed request dense as
    /// [`KvCacheFallbackReason::ReaderUnavailable`].
    pub fn expected_on(&self, reader: KvReader) -> KvPolicyDecision {
        match (self.expected, reader) {
            (KvPolicyDecision::Compressed(_), KvReader::Unavailable) => {
                KvPolicyDecision::Dense(KvCacheFallbackReason::ReaderUnavailable)
            }
            (expected, _) => expected,
        }
    }
}

/// The policy's fixed order, restated: policy, batch, family, prompt minimum, final-context
/// maximum, then request shape (multimodal content, a hybrid decoder), then geometry.
fn expected_decision(case: &KvPolicyCase) -> KvPolicyDecision {
    use KvCacheFallbackReason as Reason;
    if case.policy == KvCompressionPolicy::Off {
        return KvPolicyDecision::Dense(Reason::PolicyDisabled);
    }
    if case.batch != 1 {
        return KvPolicyDecision::Dense(Reason::BatchedDecode);
    }
    let rows = KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .filter(|row| Some(row.family) == case.family)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return KvPolicyDecision::Dense(Reason::UnqualifiedModel);
    }
    let final_context = case.context_tokens + u64::from(case.max_new_tokens);
    let admitting = rows.iter().find(|row| {
        case.context_tokens >= row.min_context_tokens
            && row.max_context_tokens.is_none_or(|max| final_context < max)
    });
    match admitting {
        Some(_) if case.multimodal || case.decoder == KvCaseDecoder::Hybrid => {
            KvPolicyDecision::Dense(Reason::UnsupportedRequest)
        }
        Some(_) if case.decoder == KvCaseDecoder::UnsupportedGeometry => {
            KvPolicyDecision::Dense(Reason::UnsupportedGeometry)
        }
        Some(row) => KvPolicyDecision::Compressed(row.format),
        None if rows
            .iter()
            .all(|row| case.context_tokens < row.min_context_tokens) =>
        {
            KvPolicyDecision::Dense(Reason::BelowMinimumContext)
        }
        None => KvPolicyDecision::Dense(Reason::AboveQualifiedContext),
    }
}

/// The token budgets of the table: none, and [`BUDGET`] tokens.
const BUDGET: u32 = 64;

/// Prompt points around every row's bounds — its prompt minimum, and its final-context maximum
/// reached with and without the [`BUDGET`] — plus a short and a very long prompt.
fn context_points() -> Vec<u64> {
    let budget = u64::from(BUDGET);
    let mut points = vec![1, 1 << 24];
    for row in KV_COMPRESSION_QUALIFICATIONS {
        points.extend([
            row.min_context_tokens.saturating_sub(1),
            row.min_context_tokens,
            row.min_context_tokens + 1,
        ]);
        if let Some(max) = row.max_context_tokens {
            points.extend([max - budget - 1, max - budget, max - 1, max]);
        }
    }
    points.sort_unstable();
    points.dedup();
    points
}

/// The fixture table: every combination of policy, table family (and no family), loaded decoder,
/// the prompt points around the qualification rows' bounds, a zero and a 64-token budget, batch 1
/// and 2, and text-only and multimodal requests.
pub fn kv_policy_cases() -> Vec<KvPolicyCase> {
    let mut cases = Vec::new();
    for policy in [KvCompressionPolicy::Off, KvCompressionPolicy::Qualified] {
        for family in [None, Some(KvModelFamily::Llama), Some(KvModelFamily::Qwen3)] {
            for decoder in KvCaseDecoder::ALL {
                for &context_tokens in &context_points() {
                    for max_new_tokens in [0, BUDGET] {
                        for batch in [1, 2] {
                            for multimodal in [false, true] {
                                let mut case = KvPolicyCase {
                                    policy,
                                    family,
                                    decoder,
                                    context_tokens,
                                    max_new_tokens,
                                    batch,
                                    multimodal,
                                    expected: KvPolicyDecision::Dense(
                                        KvCacheFallbackReason::PolicyDisabled,
                                    ),
                                };
                                case.expected = expected_decision(&case);
                                cases.push(case);
                            }
                        }
                    }
                }
            }
        }
    }
    cases
}

/// Why `actual` is not what a backend with `reader` must decide for `case`, if it is not.
fn check_case(case: &KvPolicyCase, reader: KvReader, actual: &KvBackendDecision) -> Option<String> {
    match (case.expected_on(reader), actual) {
        (KvPolicyDecision::Compressed(want), KvBackendDecision::Compressed(got))
            if want == *got =>
        {
            None
        }
        (KvPolicyDecision::Dense(want), KvBackendDecision::Dense(report)) => {
            let detailed = matches!(
                want,
                KvCacheFallbackReason::UnsupportedRequest
                    | KvCacheFallbackReason::UnsupportedGeometry
                    | KvCacheFallbackReason::ReaderUnavailable
            );
            let honest = report.fallback == Some(want)
                && report.format.is_none()
                && report.counters == KvCacheCounters::default()
                && report.format_version == KV_CACHE_FORMAT_VERSION
                && (!detailed || report.detail.is_some());
            (!honest).then(|| format!("{case:?}: expected dense {want:?}, got {report:?}"))
        }
        (want, got) => Some(format!("{case:?}: expected {want:?}, got {got:?}")),
    }
}

/// Run every [`kv_policy_cases`] request through `decide` (the backend's production planner) and
/// check it against the shared decision for a backend with `reader`. Returns every mismatch.
pub fn check_kv_policy_conformance(
    reader: KvReader,
    mut decide: impl FnMut(&KvPolicyCase) -> KvBackendDecision,
) -> Result<(), String> {
    let failures = kv_policy_cases()
        .iter()
        .filter_map(|case| check_case(case, reader, &decide(case)))
        .collect::<Vec<_>>();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} compressed-KV policy case(s) diverge from the shared decision:\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

/// [`check_kv_policy_conformance`], panicking with every mismatch.
pub fn kv_policy_conformance(
    reader: KvReader,
    decide: impl FnMut(&KvPolicyCase) -> KvBackendDecision,
) {
    if let Err(failures) = check_kv_policy_conformance(reader, decide) {
        panic!("{failures}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        plan_kv_cache, plan_kv_cache_without_reader, KvAttentionGeometry,
        KvCacheFallbackReason as Reason, KvCachePlan, KvCacheRequest,
    };

    fn request(case: &KvPolicyCase) -> KvCacheRequest {
        KvCacheRequest {
            policy: case.policy,
            family: case.family,
            context_tokens: case.context_tokens,
            max_new_tokens: u64::from(case.max_new_tokens),
            batch: case.batch,
            unsupported_request: if case.multimodal {
                Some("multimodal".into())
            } else if case.decoder == KvCaseDecoder::Hybrid {
                Some("hybrid".into())
            } else {
                None
            },
            geometry: KvAttentionGeometry {
                head_dim: if case.decoder == KvCaseDecoder::UnsupportedGeometry {
                    96
                } else {
                    128
                },
                ..KvAttentionGeometry::default()
            },
        }
    }

    fn fused(case: &KvPolicyCase) -> KvBackendDecision {
        match plan_kv_cache(request(case), |row| Ok::<_, (Reason, String)>(row.format)) {
            KvCachePlan::Compressed { reader, .. } => KvBackendDecision::Compressed(reader),
            KvCachePlan::Dense(report) => KvBackendDecision::Dense(report),
        }
    }

    fn without_reader(case: &KvPolicyCase) -> KvBackendDecision {
        KvBackendDecision::Dense(plan_kv_cache_without_reader(request(case), "test"))
    }

    /// The shared core-llm planner is the reference both backends call: it passes the fixture
    /// with and without a reader.
    #[test]
    fn the_shared_planner_conforms_with_and_without_a_reader() {
        check_kv_policy_conformance(KvReader::Fused, fused).unwrap();
        check_kv_policy_conformance(KvReader::Unavailable, without_reader).unwrap();
    }

    /// The table reaches every decision a reader-less backend can report, and every one a fused
    /// backend can (bar the reader's own geometry/reader/runtime refusals).
    #[test]
    fn the_table_covers_every_policy_decision() {
        let cases = kv_policy_cases();
        for reader in [KvReader::Fused, KvReader::Unavailable] {
            let reached = cases
                .iter()
                .map(|case| case.expected_on(reader))
                .collect::<Vec<_>>();
            let mut wanted = vec![
                KvPolicyDecision::Dense(Reason::PolicyDisabled),
                KvPolicyDecision::Dense(Reason::BatchedDecode),
                KvPolicyDecision::Dense(Reason::UnqualifiedModel),
                KvPolicyDecision::Dense(Reason::BelowMinimumContext),
                KvPolicyDecision::Dense(Reason::AboveQualifiedContext),
                KvPolicyDecision::Dense(Reason::UnsupportedRequest),
                KvPolicyDecision::Dense(Reason::UnsupportedGeometry),
            ];
            wanted.push(match reader {
                KvReader::Fused => {
                    KvPolicyDecision::Compressed(KvCompressionFormat::GroupAffineK8V8)
                }
                KvReader::Unavailable => KvPolicyDecision::Dense(Reason::ReaderUnavailable),
            });
            for decision in wanted {
                assert!(reached.contains(&decision), "{reader:?} {decision:?}");
            }
            if reader == KvReader::Unavailable {
                assert!(reached
                    .iter()
                    .all(|decision| matches!(decision, KvPolicyDecision::Dense(_))));
            }
        }
        // For every bounded row, some prompt is compressed with no budget and above the
        // qualified range with one: the token budget, not only the prompt, is under test.
        for row in KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .filter(|row| row.max_context_tokens.is_some())
        {
            let flipped = cases.iter().any(|unbudgeted| {
                unbudgeted.family == Some(row.family)
                    && unbudgeted.max_new_tokens == 0
                    && matches!(unbudgeted.expected, KvPolicyDecision::Compressed(_))
                    && cases.iter().any(|budgeted| {
                        budgeted.max_new_tokens > 0
                            && budgeted.expected
                                == KvPolicyDecision::Dense(Reason::AboveQualifiedContext)
                            && KvPolicyCase {
                                max_new_tokens: 0,
                                expected: unbudgeted.expected,
                                ..budgeted.clone()
                            } == *unbudgeted
                    })
            });
            assert!(flipped, "{:?}", row.family);
        }
    }

    /// The checker fails a backend that compresses without a reader, reports the wrong reason,
    /// claims compressed counters or a format on a dense run, or omits the detail of a backend
    /// refusal.
    #[test]
    fn the_checker_rejects_divergent_or_dishonest_backends() {
        // A reader-less backend that compresses where a fused one would.
        assert!(check_kv_policy_conformance(KvReader::Unavailable, fused).is_err());
        assert!(check_kv_policy_conformance(KvReader::Fused, without_reader).is_err());
        let mutate = |edit: fn(&mut KvCacheReport)| {
            move |case: &KvPolicyCase| match without_reader(case) {
                KvBackendDecision::Dense(mut report) => {
                    edit(&mut report);
                    KvBackendDecision::Dense(report)
                }
                compressed => compressed,
            }
        };
        let edits: [fn(&mut KvCacheReport); 5] = [
            |report| report.fallback = Some(Reason::RuntimeFallback),
            |report| report.format = Some(KvCompressionFormat::GroupAffineK8V8),
            |report| report.counters.compressed_cache_bytes = 1,
            |report| report.format_version += 1,
            |report| report.detail = None,
        ];
        for edit in edits {
            assert!(
                check_kv_policy_conformance(KvReader::Unavailable, mutate(edit)).is_err(),
                "a dishonest dense report passed"
            );
        }
    }
}
