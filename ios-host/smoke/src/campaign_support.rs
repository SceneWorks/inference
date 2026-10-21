use super::*;

/// Output cardinality, NOT the solver's U-Net batch: count is serial and repeats every prompt row.
pub(super) fn expected_generation_outputs(request: &GenerationRequest) -> Result<usize, String> {
    if request.count == 0 {
        return Err("generation count must be positive".into());
    }
    usize::try_from(request.count)
        .map_err(|e| e.to_string())?
        .checked_mul(request.prompt_batch.len().max(1))
        .ok_or_else(|| "generation output cardinality overflow".into())
}

/// Typed refusal kinds straight from runtime-ios; no message is parsed. Failures (inventory,
/// binding, malformed request, provider errors) stay diagnostic failures.
pub(super) fn classify(
    kind: &mut Option<&'static str>,
    error: runtime_ios::media::AdmissionError,
) -> AdmissionError {
    match error {
        runtime_ios::media::AdmissionError::Refused(refusal) => {
            *kind = Some(refusal.kind.label());
            AdmissionError::Refused(refusal.reason)
        }
        runtime_ios::media::AdmissionError::Failed(error) => {
            AdmissionError::Failure(error.to_string())
        }
    }
}

pub(super) fn case_timeout() -> Result<u64, String> {
    let seconds = std::env::var("IOS_CAMPAIGN_TIMEOUT_SECONDS")
        .unwrap_or_else(|_| "1800".into())
        .parse::<u64>()
        .map_err(|_| "invalid device deadline")?;
    if !(1..=20000).contains(&seconds) {
        return Err("device deadline outside finite domain".into());
    }
    Ok(seconds)
}

/// Mirror of the iPhoneOS/macOS SDK `struct task_vm_info` (`mach/task_info.h`, declared under
/// `#pragma pack(push, 4)`), through REV7. Its layout is the source of every word offset and count
/// below; `task_vm_info_layout_matches_the_sdk_header` checks it against the installed SDK header.
#[repr(C, packed(4))]
#[allow(dead_code)]
pub(super) struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    // rev1
    phys_footprint: u64,
    // rev2
    min_address: u64,
    max_address: u64,
    // rev3
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_novolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
    // rev4
    limit_bytes_remaining: u64,
    // rev5
    decompressions: i32,
    // rev6
    ledger_swapins: i64,
    // rev7
    ledger_tag_neural_nofootprint_total: i64,
    ledger_tag_neural_nofootprint_peak: i64,
}

const WORD: usize = std::mem::size_of::<u32>();
/// `TASK_VM_INFO_COUNT`: the full struct, in `natural_t` words. This is what is requested.
pub(super) const TASK_VM_INFO_COUNT: u32 = (std::mem::size_of::<TaskVmInfo>() / WORD) as u32;
/// `TASK_VM_INFO_REV1_COUNT`: the smallest reply carrying `phys_footprint`.
pub(super) const TASK_VM_INFO_REV1_COUNT: u32 =
    (std::mem::offset_of!(TaskVmInfo, min_address) / WORD) as u32;
/// `TASK_VM_INFO_REV3_COUNT`: the smallest reply carrying `ledger_phys_footprint_peak`.
pub(super) const TASK_VM_INFO_REV3_COUNT: u32 =
    (std::mem::offset_of!(TaskVmInfo, limit_bytes_remaining) / WORD) as u32;
const PHYS_FOOTPRINT_WORD: usize = std::mem::offset_of!(TaskVmInfo, phys_footprint) / WORD;
const LEDGER_PEAK_WORD: usize = std::mem::offset_of!(TaskVmInfo, ledger_phys_footprint_peak) / WORD;

/// The kernel lifetime footprint peak, or why it is absent. Never a silent `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PeakReading {
    Read(u64),
    /// The kernel replied with fewer words than `TASK_VM_INFO_REV3_COUNT`.
    Unavailable {
        returned_words: u32,
    },
}

/// The injected query keeps the exact request count and field decoding testable without calling
/// an OS or accelerator. `None` only when the call failed or the reply lacks `phys_footprint`.
pub(super) fn footprint_snapshot(
    query: impl FnOnce(&mut [u32], &mut u32) -> i32,
) -> runtime_ios::media::evidence::KernelFootprintSnapshot {
    use runtime_ios::media::evidence::{KernelFootprintMethod, KernelFootprintSnapshot};
    let mut words = [0u32; TASK_VM_INFO_COUNT as usize];
    let mut count = TASK_VM_INFO_COUNT;
    let status = query(&mut words, &mut count);
    let current_readable =
        status == 0 && (TASK_VM_INFO_REV1_COUNT..=TASK_VM_INFO_COUNT).contains(&count);
    let word64 = |at: usize| u64::from(words[at]) | (u64::from(words[at + 1]) << 32);
    KernelFootprintSnapshot {
        method: KernelFootprintMethod::TaskVmInfo,
        query_status: status,
        requested_words: TASK_VM_INFO_COUNT,
        returned_words: count,
        current_required_words: TASK_VM_INFO_REV1_COUNT,
        peak_required_words: TASK_VM_INFO_REV3_COUNT,
        current_bytes: current_readable.then(|| word64(PHYS_FOOTPRINT_WORD)),
        lifetime_peak_bytes: (current_readable && count >= TASK_VM_INFO_REV3_COUNT)
            .then(|| word64(LEDGER_PEAK_WORD)),
    }
}

#[cfg(test)]
pub(super) fn read_footprint(
    query: impl FnOnce(&mut [u32], &mut u32) -> i32,
) -> Option<(u64, PeakReading)> {
    let snapshot = footprint_snapshot(query);
    let current = snapshot.current_bytes?;
    Some((
        current,
        snapshot.lifetime_peak_bytes.map_or(
            PeakReading::Unavailable {
                returned_words: snapshot.returned_words,
            },
            PeakReading::Read,
        ),
    ))
}

pub(super) struct Deadline {
    stop: Option<std::sync::mpsc::Sender<()>>,
    pub(super) thread: Option<std::thread::JoinHandle<()>>,
}
impl Deadline {
    pub(super) fn start(cancel: CancelFlag, duration: Duration) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            if matches!(
                rx.recv_timeout(duration),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                cancel.cancel();
            }
        });
        Self {
            stop: Some(tx),
            thread: Some(thread),
        }
    }
}
impl Drop for Deadline {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Persist case outputs. `edge` is the expected square geometry, or `None` for a task whose output
/// size is the provider's (restoration's model-default scale).
pub(super) fn persist_images(
    docs: &Path,
    c: &Case,
    images: &[Image],
    edge: Option<u32>,
) -> Result<(), String> {
    for (index, image) in images.iter().enumerate() {
        if edge.is_some_and(|edge| image.width != edge || image.height != edge)
            || image.width == 0
            || image.height == 0
            || image.pixels.len() != image.width as usize * image.height as usize * 3
            || image.pixels.iter().min() == image.pixels.iter().max()
        {
            return Err("wrong geometry or constant image".into());
        }
        let name = format!(
            "campaign-{}-{index}.png",
            runtime_ios::media::capture_artifact_stem(&c.key())
        );
        image::save_buffer_with_format(
            docs.join(&name),
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut manifest = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(docs.join(crate::RENDERS_MANIFEST))
            .map_err(|e| e.to_string())?;
        writeln!(manifest, "{name}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(super) fn growth_report(
    growth: &Result<mlx_gen::ios_anchor_law::KernelGrowth, String>,
) -> Value {
    use mlx_gen::ios_anchor_law::KernelPeakQuality;
    match growth {
        Ok(growth) => json!({"available":true,"growth_upper_bytes":growth.growth_upper,
            "quality":match growth.quality { KernelPeakQuality::AdvancedDuringRun=>"advanced_during_run",
                KernelPeakQuality::CurrentAttainedUpper=>"current_attained_upper", KernelPeakQuality::MaskedUpper=>"masked_upper" },
            "exact_growth_bytes":growth.exact_growth().ok(), "valid_query_count":growth.query_count}),
        Err(reason) => json!({"available":false,"reason":reason}),
    }
}

/// Labelled TOOLCHAIN-LAYOUT assumptions (the authority-pinned Rust 1.96 std, ac68), NOT campaign
/// source: (a) `str::to_string` / `String::from` / `&str -> String` allocate exactly `len` bytes;
/// (b) RawVec amortized growth: a Vec grown by `push`/`collect` to `n` elements allocates at most
/// `(4·n + 8)·size_of::<T>()` cumulative bytes with no drop credit; (c) an insert-only BTreeMap
/// (B = 6: 11 slots, 12 edges) allocates at most `entries + 1` nodes, each no larger than an
/// internal node (parent pointer, parent index, length, 11 keys, 11 values, 12 edges).
fn btree_node_bytes(key: usize, value: usize) -> Option<u64> {
    let leaf = 8usize
        .checked_add(4)?
        .checked_add(key.checked_mul(11)?)?
        .checked_add(value.checked_mul(11)?)?;
    let aligned = leaf.checked_add(7)? / 8 * 8;
    u64::try_from(aligned.checked_add(12 * 8)?).ok()
}

fn vec_growth_bytes(n: u64, elem: usize) -> Option<u64> {
    n.checked_mul(4)?
        .checked_add(8)?
        .checked_mul(u64::try_from(elem).ok()?)
}

fn btree_bytes(entries: u64, key: usize, value: usize) -> Option<u64> {
    entries
        .checked_add(1)?
        .checked_mul(btree_node_bytes(key, value)?)
}

/// The longest phase key `QuantumCoverage::record` allocates for a check (`IosPhase::as_key`).
fn max_phase_key_bytes() -> u64 {
    mlx_gen::ios_pricing::IosPhase::ALL
        .iter()
        .map(|phase| phase.as_key().len() as u64)
        .max()
        .unwrap_or(0)
}

/// Source-derived host-allocation upper of THIS campaign's own quantum listener
/// (`campaign_media::run_case`) and its no-op progress callbacks (`|_| {}`, non-allocating),
/// stated to admission as the caller-owned callback term. Bounded by the recorders' own caps:
/// `SamplingCoverage` retains at most `MAX_SAMPLING_MARKS` accepted marks (names ≤
/// `MAX_SAMPLING_NAME_BYTES`, ≤ `MAX_SAMPLING_FIELDS` fields) and allocates nothing for dropped
/// ones; `QuantumCoverage` retains at most `MAX_QUANTUM_CHECKS` checks. Every label is stored in
/// FULL (never capped or truncated): its copy is stated per label byte and the provider
/// multiplies by its own source label lengths. Per quantum mark it allocates one context Vec
/// (≤ one entry per accepted metadata mark), one phase-key String, one `Mark` slot and one
/// exact-length label copy. Uses the labelled toolchain-layout assumptions above.
pub(super) fn capture_listener_callback_host(
    provider: &str,
) -> Result<runtime_ios::media::IosCallbackHost, String> {
    use runtime_ios::media::evidence::{
        QuantumCheck, SamplingMark, MAX_QUANTUM_CHECKS, MAX_SAMPLING_FIELDS, MAX_SAMPLING_MARKS,
        MAX_SAMPLING_NAME_BYTES,
    };
    use std::collections::BTreeMap;
    use std::mem::size_of;
    let overflow = || "capture listener host bound overflows".to_string();
    let name = MAX_SAMPLING_NAME_BYTES as u64;
    let marks = MAX_SAMPLING_MARKS as u64;
    let fields = MAX_SAMPLING_FIELDS as u64;
    let checks = MAX_QUANTUM_CHECKS as u64;
    let sum = |terms: &[Option<u64>]| -> Result<u64, String> {
        terms.iter().try_fold(0u64, |total, term| {
            total
                .checked_add(term.ok_or_else(overflow)?)
                .ok_or_else(overflow)
        })
    };
    // One accepted metadata mark: SamplingMark provider/kind Strings and its fields map (keys and
    // nodes); `counts` and `latest` allocate a provider and kind String per accepted call.
    let per_accepted = sum(&[
        name.checked_mul(2),
        fields.checked_mul(name),
        btree_bytes(fields, size_of::<String>(), size_of::<u64>()),
        name.checked_mul(2),
        name.checked_mul(2),
    ])?;
    let fixed = sum(&[
        // Installation: the provider String clone, both Arcs and the boxed closure (two Arcs
        // and one String captured).
        Some(provider.len() as u64),
        Some((size_of::<std::sync::Mutex<CaptureMarks>>() + 16) as u64),
        Some((size_of::<std::sync::atomic::AtomicBool>() + 16) as u64),
        Some((2 * size_of::<usize>() + size_of::<String>()) as u64),
        marks.checked_mul(per_accepted),
        vec_growth_bytes(marks, size_of::<SamplingMark>()),
        // `counts` / `latest`: outer maps (≤ one entry per accepted mark) and inner maps
        // (≤ one node per accepted entry plus one root per inner map).
        btree_bytes(
            marks,
            size_of::<String>(),
            size_of::<BTreeMap<String, u64>>(),
        ),
        marks
            .checked_mul(2)
            .and_then(|nodes| nodes.checked_mul(btree_node_bytes(size_of::<String>(), 8)?)),
        btree_bytes(
            marks,
            size_of::<String>(),
            size_of::<BTreeMap<String, u32>>(),
        ),
        marks
            .checked_mul(2)
            .and_then(|nodes| nodes.checked_mul(btree_node_bytes(size_of::<String>(), 4)?)),
        // Retained checks: each new check's label String plus the checks Vec growth.
        checks.checked_mul(name),
        vec_growth_bytes(checks, size_of::<QuantumCheck>()),
        // The constant part of the `marks` Vec growth.
        vec_growth_bytes(0, size_of::<Mark>()),
    ])?;
    let per_quantum_mark = sum(&[
        vec_growth_bytes(marks, size_of::<u32>()),
        Some(max_phase_key_bytes()),
        (size_of::<Mark>() as u64).checked_mul(4),
    ])?;
    Ok(runtime_ios::media::IosCallbackHost::Bounded {
        fixed_bytes: fixed,
        per_progress_event_bytes: 0,
        per_preview_event_bytes: 0,
        per_quantum_mark_bytes: per_quantum_mark,
        // `label.to_string()`: exactly the label's bytes (toolchain assumption (a)).
        per_quantum_label_byte_bytes: 1,
    })
}

#[derive(Default)]
pub(super) struct CaptureMarks {
    pub marks: Vec<Mark>,
    pub sampling: runtime_ios::media::evidence::SamplingCoverage,
    pub quantum: runtime_ios::media::evidence::QuantumCoverage,
    pub preceding_sampling_mark: Option<u32>,
    pub quantum_query_gaps: u64,
    pub model_mark_start: Option<usize>,
    latest: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u32>>,
}

impl CaptureMarks {
    pub fn metadata(&mut self, provider: &str, kind: &str, fields: &[(&str, u64)]) {
        if provider == "ios_capture" && kind == "before-model-load" {
            self.model_mark_start = Some(self.marks.len());
        }
        self.preceding_sampling_mark = self.sampling.record(provider, kind, fields);
        if let Some(index) = self.preceding_sampling_mark {
            self.latest
                .entry(provider.into())
                .or_default()
                .insert(kind.into(), index);
        } else if let Some(kinds) = self.latest.get_mut(provider) {
            kinds.remove(kind);
        }
    }

    pub fn checked(&mut self, phase: Option<mlx_gen::ios_pricing::IosPhase>, label: &str) {
        let context = self
            .latest
            .values()
            .flat_map(|kinds| kinds.values().copied())
            .collect();
        self.quantum.record(phase, label, context);
    }
}

/// One quantum boundary seen by a capture: the phase its label runs in (the provider's own
/// `ios_quantum_phase`; `None` for a label it does not attribute), whether it OPENS a quantum (a
/// pre-construction check) or closes one (a synchronization), and the kernel lifetime
/// `phys_footprint` peak at that instant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Mark {
    pub phase: Option<mlx_gen::ios_pricing::IosPhase>,
    pub opens: bool,
    pub lifetime_peak: u64,
    /// The quantum's label, its priced scratch (0 for a synchronization) and the CURRENT
    /// `phys_footprint` at the mark.
    pub label: String,
    pub scratch: u64,
    pub current: u64,
    pub preceding_sampling_mark: Option<u32>,
}

/// One quantum formula as the device saw it: the priced scratch and the observed growth of the
/// invocation with the largest observed/priced ratio, and whether that growth is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Observation {
    pub priced: u64,
    pub observed: u64,
    pub exact: bool,
    pub preceding_sampling_mark: Option<u32>,
}

/// The per-quantum DEVICE observations of a capture: for every quantum a check opened, the
/// kernel-footprint growth across it — the lifetime peak at the next mark minus the footprint at
/// the check. EXACT when the lifetime peak rose inside the quantum (so that reading is the
/// quantum's own peak); otherwise an earlier peak masked it and the reading is an upper bound. Per
/// label the invocation with the largest observed/priced ratio is kept (an exact one before any
/// masked one): the formula's worst under-pricing on the device.
pub(super) fn quantum_observations(
    marks: &[Mark],
    end_peak: u64,
) -> std::collections::BTreeMap<String, Observation> {
    let mut out: std::collections::BTreeMap<String, Observation> = Default::default();
    for (i, mark) in marks.iter().enumerate() {
        if !mark.opens || mark.scratch == 0 {
            continue;
        }
        let after = marks.get(i + 1).map_or(end_peak, |next| next.lifetime_peak);
        let exact = after > mark.lifetime_peak;
        let observed = if exact { after } else { mark.lifetime_peak }.saturating_sub(mark.current);
        let candidate = Observation {
            priced: mark.scratch,
            observed,
            exact,
            preceding_sampling_mark: mark.preceding_sampling_mark,
        };
        let better = out.get(&mark.label).is_none_or(|existing| {
            (candidate.exact && !existing.exact)
                || (candidate.exact == existing.exact
                    && u128::from(candidate.observed) * u128::from(existing.priced)
                        > u128::from(existing.observed) * u128::from(candidate.priced))
        });
        if better {
            out.insert(mark.label.clone(), candidate);
        }
    }
    out
}

/// Split a run's kernel lifetime peak by phase through its quantum marks.
///
/// The lifetime peak is monotone, so the growth between two consecutive readings happened in the
/// interval between them. That interval belongs to the quantum the earlier mark OPENED, else (glue
/// after a synchronization) to the quantum the later mark belongs to; the stretch before the first
/// mark to the first mark's phase and the stretch after the last to the last mark's. A phase's peak
/// is the largest lifetime reading at the end of its intervals; it is `exact` when that reading was
/// reached by growth inside the phase's own interval (otherwise an earlier phase masked it and the
/// reading is an upper bound). Unattributed intervals are dropped (their phase stays absent and is
/// priced at the whole-run peak downstream, erring large).
pub(super) fn phase_peaks(
    start_peak: u64,
    marks: &[Mark],
    end_peak: u64,
) -> std::collections::BTreeMap<mlx_gen::ios_pricing::IosPhase, (u64, bool)> {
    let mut out: std::collections::BTreeMap<_, (u64, bool)> = Default::default();
    let mut record = |phase: Option<_>, before: u64, after: u64| {
        let Some(phase) = phase else { return };
        let slot = out.entry(phase).or_insert((0, false));
        if after > slot.0 {
            *slot = (after, after > before);
        } else if after == slot.0 && after > before {
            slot.1 = true;
        }
    };
    let mut previous = start_peak;
    for (i, mark) in marks.iter().enumerate() {
        let phase = match i.checked_sub(1).map(|j| &marks[j]) {
            Some(earlier) if earlier.opens => earlier.phase,
            _ => mark.phase,
        };
        record(phase, previous, mark.lifetime_peak);
        previous = previous.max(mark.lifetime_peak);
    }
    if let Some(last) = marks.last() {
        record(last.phase, previous, end_peak.max(previous));
    }
    out
}
