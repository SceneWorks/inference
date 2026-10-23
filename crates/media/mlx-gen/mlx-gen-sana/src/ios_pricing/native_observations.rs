//! Actual receipt chronology for the CLOSED SANA program. Metadata selects a
//! raw projection only; full source/count validation below is the authority.
//! A native captured-prefix receipt is NOT exclusive request ownership, alias
//! death, allocator-cache release, or process-footprint release.
use super::super::native_graph::{native_epoch_labels, NativeEpochCounts};
use super::*;

/// Borrowed recorder adapters; neither view owns/clones metadata or media.
#[derive(Clone, Copy)]
pub struct RetirementMetadataView<'a> {
    pub provider: &'a str,
    pub kind: &'a str,
    pub fields: &'a BTreeMap<String, u64>,
}
#[derive(Clone, Copy)]
pub struct RetirementCheckView<'a> {
    pub phase: Option<IosPhase>,
    pub label: &'a str,
    pub context: &'a [u32],
    pub count: u64,
}
trait ScalarMetadata {
    fn scalar_fields(&self) -> &BTreeMap<String, u64>;
}
impl ScalarMetadata for RetirementMetadataView<'_> {
    fn scalar_fields(&self) -> &BTreeMap<String, u64> {
        self.fields
    }
}
impl ScalarMetadata for &mlx_gen::ios_pricing::IosObservedMetadata {
    fn scalar_fields(&self) -> &BTreeMap<String, u64> {
        &self.fields
    }
}
fn field(mark: impl ScalarMetadata, name: &str) -> Result<u64> {
    mark.scalar_fields()
        .get(name)
        .copied()
        .ok_or_else(|| invalid("retirement observation missing scalar"))
}
fn is_load(label: &str) -> bool {
    matches!(
        label,
        labels::TEXT_LOAD | labels::REFERENCE_ENCODER_LOAD | labels::HEAVY_LOAD
    )
}
/// Full actual receipt chronology, shared by native certificate observations
/// and completed-capture execution identity validation. This ALONE grants no
/// native bound or physical permission; exact source multiplicities follow.
pub fn validate_retirement_receipts<'a>(
    metadata_len: usize,
    metadata: impl Fn(usize) -> Option<RetirementMetadataView<'a>>,
    checks: impl Iterator<Item = RetirementCheckView<'a>>,
    complete: bool,
) -> Result<()> {
    if !complete {
        return Err(invalid("retirement observation is truncated"));
    }
    let mut start = None;
    let mut end = None;
    let mut emitted = [0u64; 5];
    for index in 0..metadata_len {
        let mark = metadata(index).ok_or_else(|| invalid("retirement metadata index missing"))?;
        let category = match (mark.provider, mark.kind) {
            ("mlx", "native-quantum") => Some(0),
            ("mlx", "native-joined") => Some(1),
            ("sana", "native-dcae-glu") => Some(2),
            ("sana", "native-selected") => Some(3),
            ("sana", "native-owned-phase-retired") => Some(4),
            _ => None,
        };
        if let Some(category) = category {
            emitted[category] = emitted[category]
                .checked_add(1)
                .ok_or_else(|| invalid("native metadata emitter count overflow"))?;
        }
        if mark.kind == "native-lifecycle" && start.replace((index, mark)).is_some() {
            return Err(invalid("multiple native invocation starts"));
        }
        if mark.kind == "native-lifecycle-completed" && end.replace((index, mark)).is_some() {
            return Err(invalid("multiple native invocation completions"));
        }
    }
    let (start_index, start) =
        start.ok_or_else(|| invalid("missing actual configured invocation"))?;
    let (end_index, end) =
        end.ok_or_else(|| invalid("missing actual successful invocation completion"))?;
    if !matches!(start.provider, crate::MODEL_ID | crate::SPRINT_MODEL_ID)
        || start.provider != end.provider
        || start_index >= end_index
        || !fields_match(
            start.fields,
            &[
                ("schema", 1),
                ("started", 1),
                ("configured", 1),
                ("epoch", 0),
            ],
        )
        || !fields_match(end.fields, &[("schema", 1), ("status", 1)])
    {
        return Err(invalid(
            "native lifecycle was not actually configured/completed",
        ));
    }
    let invocation = field(start, "invocation")?;
    if invocation == 0 || field(end, "invocation")? != invocation {
        return Err(invalid("native invocation identity mismatch"));
    }
    let mask = field(start, "stream_mask")?;
    if mask == 0
        || mask & !7 != 0
        || field(start, "stream_count")? != u64::from(mask.count_ones())
        || field(start, "compute_cpu")? > 1
        || !fields_match(
            start.fields,
            &[
                ("exclusive_request_ownership", 0),
                ("host_reserve_declared", 0),
            ],
        )
    {
        return Err(invalid(
            "native source stream-set/ownership metadata invalid",
        ));
    }
    let stream_keys = ["stream_0", "stream_1", "stream_2"];
    let selected = field(start, "selected_stream")?;
    let mut selected_seen = false;
    for (index, key) in stream_keys.into_iter().enumerate() {
        if mask & (1 << index) == 0 {
            continue;
        }
        let stream = field(start, key)?;
        selected_seen |= stream == selected;
        for (previous, key) in stream_keys.iter().enumerate().take(index) {
            if mask & (1 << previous) != 0 && field(start, key)? == stream {
                return Err(invalid("native stream set contains duplicate handles"));
            }
        }
    }
    if !selected_seen {
        return Err(invalid(
            "selected compute stream omitted from retired ranges",
        ));
    }
    let mut epoch = 0u64;
    let mut expected_joined = 0u64;
    let mut expected_glu = 0u64;
    let mut expected_selected = 0u64;
    for check in checks {
        if check.count == 0 {
            return Err(invalid("zero-count native checked witness"));
        }
        if !native_epoch_labels().any(|known| known == check.label) {
            return Err(invalid("foreign quantum in closed SANA invocation"));
        }
        if check.phase != ios_quantum_phase(check.label) {
            return Err(invalid("native source quantum phase changed"));
        }
        let mut actual = None;
        for &index in check.context {
            let index = index as usize;
            let mark = metadata(index).ok_or_else(|| invalid("retirement context out of range"))?;
            if mark.provider == "mlx"
                && mark.kind == "native-quantum"
                && actual.replace((index, mark)).is_some()
            {
                return Err(invalid("duplicate actual native epoch context"));
            }
        }
        let (index, mark) = match actual {
            Some(actual) => actual,
            // Eager model loading occurs before the generation invocation.
            // Its backing is NEVER retired by the tight source model, so it
            // needs ordinary source/page observations but no invented epoch.
            None if is_load(check.label) => continue,
            None => return Err(invalid("checked quantum has no actual invocation epoch")),
        };
        if check.count != 1 {
            return Err(invalid(
                "native epochs cannot be compressed across distinct receipts",
            ));
        }
        epoch = epoch
            .checked_add(1)
            .ok_or_else(|| invalid("native epoch count overflow"))?;
        if index <= start_index
            || index >= end_index
            || !fields_match(
                mark.fields,
                &[
                    ("schema", 1),
                    ("invocation", invocation),
                    ("epoch", epoch),
                    (
                        "label",
                        mlx_gen::ios_quantum::retirement_label_id(check.label),
                    ),
                ],
            )
        {
            return Err(invalid("actual native checked epoch changed/order gap"));
        }
        // Loads grant no retirement credit: their complete check backing is
        // conservatively held by the source ledger for the entire request.
        if is_load(check.label) {
            continue;
        }
        expected_joined = expected_joined
            .checked_add(1)
            .ok_or_else(|| invalid("native joined count overflow"))?;
        if matches!(check.label, labels::ENCODER_GLU | labels::DECODER_GLU) {
            expected_glu = expected_glu
                .checked_add(1)
                .ok_or_else(|| invalid("native GLU count overflow"))?;
        }
        if check.label == labels::TEXT_SELECTED {
            expected_selected = expected_selected
                .checked_add(1)
                .ok_or_else(|| invalid("native selection count overflow"))?;
        }
        let mut joined = None;
        for candidate_index in index + 1..end_index {
            let candidate = metadata(candidate_index)
                .ok_or_else(|| invalid("retirement receipt metadata missing"))?;
            if candidate.provider == "mlx" && candidate.kind == "native-quantum" {
                break;
            }
            if candidate.provider == "mlx"
                && candidate.kind == "native-joined"
                && joined.replace((candidate_index, candidate)).is_some()
            {
                return Err(invalid("duplicate native joined receipt"));
            }
        }
        let (_, joined) = joined.ok_or_else(|| {
            invalid("checked source quantum lacks successful explicit native range")
        })?;
        if !fields_match(
            joined.fields,
            &[
                ("schema", 1),
                ("invocation", invocation),
                ("epoch", epoch),
                (
                    "label",
                    mlx_gen::ios_quantum::retirement_label_id(check.label),
                ),
                ("all_source_streams", 1),
                ("status", 1),
            ],
        ) {
            return Err(invalid(
                "native receipt belongs to another epoch/range or failed",
            ));
        }
        for key in [
            "selected_stream",
            "stream_mask",
            "stream_count",
            "stream_0",
            "stream_1",
            "stream_2",
            "compute_cpu",
        ] {
            if field(joined, key)? != field(start, key)? {
                return Err(invalid(
                    "native receipt stream set changed during invocation",
                ));
            }
        }
        if !fields_match(
            joined.fields,
            &[
                ("range_captured_prefix", 1),
                ("exclusive_request_ownership", 0),
                ("alias_release_credit", 0),
                ("cache_release_credit", 0),
                ("process_release_credit", 0),
                ("caller_receipts_live_upper", 1),
                ("host_reserve_declared", 0),
            ],
        ) {
            return Err(invalid(
                "receipt claimed unsupported alias/cache/host/process release",
            ));
        }
    }
    if emitted[0] != epoch
        || emitted[1] != expected_joined
        || emitted[2] != expected_glu
        || emitted[3] != expected_selected
        || emitted[4] > 1
    {
        return Err(invalid(
            "native emitter metadata cardinality differs from actual checked programme",
        ));
    }
    if epoch == 0 || field(end, "epochs")? != epoch {
        return Err(invalid("native invocation completion omitted checked work"));
    }
    // Hashes are scalar chronology only; prove no collision in this closed
    // source label universe instead of granting arbitrary labels authority.
    for (index, label) in native_epoch_labels().enumerate() {
        if native_epoch_labels().take(index).any(|other| {
            other != label
                && mlx_gen::ios_quantum::retirement_label_id(other)
                    == mlx_gen::ios_quantum::retirement_label_id(label)
        }) {
            return Err(invalid("closed native label identities collide"));
        }
    }
    Ok(())
}

fn chronology(capture: &IosAnchorObservation) -> Result<()> {
    validate_retirement_receipts(
        capture.metadata.len(),
        |index| {
            capture
                .metadata
                .get(index)
                .map(|mark| RetirementMetadataView {
                    provider: &mark.provider,
                    kind: &mark.kind,
                    fields: &mark.fields,
                })
        },
        capture.checks.iter().map(|check| RetirementCheckView {
            phase: check.phase,
            label: &check.label,
            context: &check.context,
            count: 1,
        }),
        capture.telemetry_complete,
    )
}

/// Projection selection only. Missing/failed/truncated marks keep Ordinary
/// pricing and valid ordinary evidence; they NEVER manufacture tight authority.
pub(super) fn observed_regime(pricing: &IosPricing) -> SanaPricingRegime {
    if !pricing.observations.is_empty()
        && pricing
            .observations
            .iter()
            .all(|capture| chronology(capture).is_ok())
    {
        SanaPricingRegime::RetiredLifecycle
    } else {
        SanaPricingRegime::Ordinary
    }
}

pub(super) fn validate(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    pricing: &IosPricing,
    counts: &NativeEpochCounts,
    guided: bool,
) -> Result<()> {
    if pricing.observations.is_empty() {
        return Err(invalid("no actual native lifecycle observations"));
    }
    for capture in &pricing.observations {
        chronology(capture)?;
        let mut reference_retirements = 0u64;
        for (index, mark) in capture.metadata.iter().enumerate() {
            if mark.provider != "sana" || mark.kind != "native-owned-phase-retired" {
                continue;
            }
            reference_retirements += 1;
            let previous = capture.metadata[..index]
                .iter()
                .rev()
                .find(|m| m.provider == "mlx" && m.kind == "native-quantum")
                .ok_or_else(|| invalid("owned phase retired without actual preceding work"))?;
            if !fields_match(
                &mark.fields,
                &[
                    ("schema", 1),
                    ("component", 1),
                    ("prefix_complete", 1),
                    ("rust_drop_pending", 1),
                    ("external_alias_release_credit", 0),
                    ("host_reserve_declared", 0),
                    ("invocation", field(previous, "invocation")?),
                    ("after_epoch", field(previous, "epoch")?),
                ],
            ) {
                return Err(invalid(
                    "staged encoder ownership was not closed by its actual captured prefix",
                ));
            }
            let heavy = capture
                .checks
                .iter()
                .find(|check| check.label == labels::HEAVY_LOAD)
                .ok_or_else(|| invalid("staged heavy load missing"))?;
            let heavy_index = heavy
                .context
                .iter()
                .copied()
                .find(|&index| {
                    let mark = &capture.metadata[index as usize];
                    mark.provider == "mlx" && mark.kind == "native-quantum"
                })
                .ok_or_else(|| invalid("staged heavy load epoch missing"))?;
            if index >= heavy_index as usize {
                return Err(invalid("encoder release was not before heavy model load"));
            }
        }
        if reference_retirements != u64::from(plan.sequential && plan.img2img) {
            return Err(invalid(
                "actual staged reference encoder retirement cardinality changed",
            ));
        }
        for label in native_epoch_labels() {
            if is_load(label) || matches!(label, labels::LATENT | labels::LATENT_BATCH) {
                continue;
            }
            let actual = capture
                .checks
                .iter()
                .filter(|check| check.label == label)
                .count() as u64;
            if actual != counts.count(label)? {
                return Err(invalid(&format!("actual {label} epoch cardinality differs from exact source solver/tile program")));
            }
        }
        for check in &capture.checks {
            if matches!(
                check.label.as_str(),
                labels::DECODER_GLU | labels::ENCODER_GLU
            ) {
                let fields = context(capture, check, "sana", "native-dcae-glu")?;
                let extent = ["batch", "height", "width", "channels"]
                    .into_iter()
                    .try_fold(1u64, |size, key| {
                        size.checked_mul(
                            *fields
                                .get(key)
                                .ok_or_else(|| invalid("missing actual GLU shape"))?,
                        )
                        .ok_or_else(|| invalid("actual GLU shape overflow"))
                    })?;
                let source_shape =
                    facts
                        .dc_ae
                        .block_types
                        .iter()
                        .enumerate()
                        .any(|(stage, kind)| {
                            let Some(scale) = 1u64.checked_shl(stage as u32) else {
                                return false;
                            };
                            *kind == BlockType::EfficientVit
                                && fields_match(
                                    fields,
                                    &[
                                        ("batch", 1),
                                        ("height", u64::from(plan.height).div_ceil(scale)),
                                        ("width", u64::from(plan.width).div_ceil(scale)),
                                        ("channels", facts.dc_ae.block_out_channels[stage] as u64),
                                    ],
                                )
                        });
                if !source_shape
                    || !fields_match(
                        fields,
                        &[
                            ("schema", 1),
                            ("encode", u64::from(check.label == labels::ENCODER_GLU)),
                            ("original_elements", extent),
                            ("attended_elements", extent),
                        ],
                    )
                {
                    return Err(invalid(
                        "joined GLU lost caller-held original or attended feature",
                    ));
                }
            }
            if check.label == labels::TEXT_SELECTED {
                let branches = if guided { 2 } else { 1 };
                let masks = branches * facts.caption_tokens;
                let captions = masks
                    .checked_mul(facts.gemma.hidden_size as u64)
                    .ok_or_else(|| invalid("selected caption extent overflow"))?;
                if !fields_match(
                    context(capture, check, "sana", "native-selected")?,
                    &[
                        ("schema", 1),
                        ("branches", branches),
                        ("caption_elements", captions),
                        ("mask_elements", masks),
                    ],
                ) {
                    return Err(invalid(
                        "selected captions do not match actual conditioning owners",
                    ));
                }
            }
        }
        let batch = capture
            .checks
            .iter()
            .filter(|check| check.label == labels::LATENT_BATCH)
            .count() as u64;
        let single = capture
            .checks
            .iter()
            .filter(|check| check.label == labels::LATENT)
            .count() as u64;
        if !((batch == 1 && single == 0) || (batch == 0 && single == u64::from(plan.count))) {
            return Err(invalid(
                "final latent ownership boundary missing or repeated",
            ));
        }
        let pixels = capture
            .checks
            .iter()
            .filter(|check| check.label == labels::DECODED_PIXELS)
            .count() as u64;
        if pixels != u64::from(plan.count) {
            return Err(invalid(
                "not every serial output reached actual decoded pixels",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Synthetic recorder protocol ONLY. These tests never obtain a native
    // certificate/physical bound; real positive bounds use actual captures.
    use super::*;
    fn mark(
        provider: &str,
        kind: &str,
        fields: &[(&str, u64)],
    ) -> mlx_gen::ios_pricing::IosObservedMetadata {
        mlx_gen::ios_pricing::IosObservedMetadata {
            provider: provider.into(),
            kind: kind.into(),
            fields: fields
                .iter()
                .map(|(key, value)| ((*key).into(), *value))
                .collect(),
        }
    }
    fn protocol() -> IosPricing {
        let cache = [
            ("selected_stream", 1),
            ("stream_mask", 2),
            ("stream_count", 1),
            ("stream_0", 0),
            ("stream_1", 1),
            ("stream_2", 0),
            ("compute_cpu", 1),
        ];
        let mut start = mark(
            crate::MODEL_ID,
            "native-lifecycle",
            &[
                ("schema", 1),
                ("invocation", 9),
                ("started", 1),
                ("configured", 1),
                ("epoch", 0),
                ("exclusive_request_ownership", 0),
                ("host_reserve_declared", 0),
            ],
        );
        start
            .fields
            .extend(cache.map(|(key, value)| (key.into(), value)));
        let pre = mark(
            "mlx",
            "native-quantum",
            &[
                ("schema", 1),
                ("invocation", 9),
                ("epoch", 1),
                (
                    "label",
                    mlx_gen::ios_quantum::retirement_label_id(GEMMA_LABELS[0]),
                ),
            ],
        );
        let mut joined = mark(
            "mlx",
            "native-joined",
            &[
                ("schema", 1),
                ("invocation", 9),
                ("epoch", 1),
                (
                    "label",
                    mlx_gen::ios_quantum::retirement_label_id(GEMMA_LABELS[0]),
                ),
                ("all_source_streams", 1),
                ("status", 1),
                ("range_captured_prefix", 1),
                ("exclusive_request_ownership", 0),
                ("alias_release_credit", 0),
                ("cache_release_credit", 0),
                ("process_release_credit", 0),
                ("caller_receipts_live_upper", 1),
                ("host_reserve_declared", 0),
            ],
        );
        joined
            .fields
            .extend(cache.map(|(key, value)| (key.into(), value)));
        let end = mark(
            crate::MODEL_ID,
            "native-lifecycle-completed",
            &[
                ("schema", 1),
                ("invocation", 9),
                ("status", 1),
                ("epochs", 1),
            ],
        );
        IosPricing {
            observations: vec![IosAnchorObservation {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                metadata: vec![start, pre, joined, end],
                checks: vec![IosObservedCheck {
                    phase: Some(IosPhase::Conditioning),
                    label: GEMMA_LABELS[0].into(),
                    context: vec![0, 1],
                }],
                telemetry_complete: true,
                ..Default::default()
            }],
            ..Default::default()
        }
    }
    #[test]
    fn missing_failed_or_truncated_receipts_keep_ordinary_projection() {
        assert_eq!(
            observed_regime(&IosPricing::default()),
            SanaPricingRegime::Ordinary
        );
        let baseline = protocol();
        assert_eq!(
            observed_regime(&baseline),
            SanaPricingRegime::RetiredLifecycle
        );
        let mut incomplete = baseline.clone();
        incomplete.observations[0].telemetry_complete = false;
        assert_eq!(observed_regime(&incomplete), SanaPricingRegime::Ordinary);
        for field in [
            "status",
            "all_source_streams",
            "range_captured_prefix",
            "epoch",
            "invocation",
            "selected_stream",
            "stream_mask",
            "stream_count",
            "compute_cpu",
        ] {
            let mut changed = baseline.clone();
            changed.observations[0].metadata[2].fields.remove(field);
            assert_eq!(
                observed_regime(&changed),
                SanaPricingRegime::Ordinary,
                "missing {field}"
            );
        }
        let mut wrong_phase = baseline;
        wrong_phase.observations[0].checks[0].phase = Some(IosPhase::Decode);
        assert_eq!(observed_regime(&wrong_phase), SanaPricingRegime::Ordinary);
    }
    #[test]
    fn another_range_or_unsupported_release_claim_never_selects_tight() {
        for field in [
            "invocation",
            "epoch",
            "selected_stream",
            "stream_1",
            "alias_release_credit",
            "cache_release_credit",
            "process_release_credit",
            "exclusive_request_ownership",
            "host_reserve_declared",
        ] {
            let mut changed = protocol();
            let value = changed.observations[0].metadata[2]
                .fields
                .get_mut(field)
                .unwrap();
            *value += 1;
            assert_eq!(
                observed_regime(&changed),
                SanaPricingRegime::Ordinary,
                "changed {field}"
            );
        }
    }
    #[test]
    fn preinvocation_load_has_no_fabricated_retirement_epoch() {
        let mut profile = protocol();
        let capture = &mut profile.observations[0];
        capture
            .metadata
            .insert(0, mark("mlx", "allocation-scope", &[("schema", 1)]));
        for check in &mut capture.checks {
            for index in &mut check.context {
                *index += 1;
            }
        }
        capture.checks.insert(
            0,
            IosObservedCheck {
                phase: Some(IosPhase::Load),
                label: labels::TEXT_LOAD.into(),
                context: vec![0],
            },
        );
        assert_eq!(
            observed_regime(&profile),
            SanaPricingRegime::RetiredLifecycle
        );
        // This is still only a topology projection, not source multiplicity,
        // counter-domain, actual model execution or certified native authority.
    }
}
