use super::*;

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
pub(super) fn read_footprint(
    query: impl FnOnce(&mut [u32], &mut u32) -> i32,
) -> Option<(u64, PeakReading)> {
    let mut words = vec![0u32; TASK_VM_INFO_COUNT as usize];
    let mut count = TASK_VM_INFO_COUNT;
    let status = query(&mut words, &mut count);
    if status != 0 || !(TASK_VM_INFO_REV1_COUNT..=TASK_VM_INFO_COUNT).contains(&count) {
        return None;
    }
    let word64 = |at: usize| u64::from(words[at]) | (u64::from(words[at + 1]) << 32);
    let current = word64(PHYS_FOOTPRINT_WORD);
    let peak = if count >= TASK_VM_INFO_REV3_COUNT {
        PeakReading::Read(word64(LEDGER_PEAK_WORD))
    } else {
        PeakReading::Unavailable {
            returned_words: count,
        }
    };
    Some((current, peak))
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
        let name = format!("campaign-{}-{index}.png", c.key().replace('/', "-"));
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
