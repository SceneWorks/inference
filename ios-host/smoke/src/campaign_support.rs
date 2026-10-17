use super::*;

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Only known evidence/capacity answers are refusals. Inventory, binding and request errors
/// remain diagnostic failures, including provider errors wrapped as Unsupported.
pub(super) fn classify_admission_error(
    kind: &mut Option<&'static str>,
    reason: String,
) -> AdmissionError {
    if reason.starts_with("unsupported: iOS SANA admission: no artifact-bound device footprint")
        || reason.ends_with("unsupported: no applicable device/request capture (source unopened)")
        || reason.ends_with("unsupported: no applicable artifact/device/request capture")
    {
        *kind = Some("no_evidence");
        AdmissionError::Refused(reason)
    } else if reason.starts_with("unsupported: iOS admission: estimated footprint")
        && reason.contains("exceeds")
    {
        *kind = Some("capacity");
        AdmissionError::Refused(reason)
    } else {
        AdmissionError::Failure(reason)
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

/// Stable task_vm_info prefix through ledger_phys_footprint_peak. The injected query keeps
/// the exact request count and field decoding testable without calling an OS or accelerator.
pub(super) fn read_footprint(
    query: impl FnOnce(&mut [u32; 44], &mut u32) -> i32,
) -> Option<(u64, Option<u64>)> {
    let mut words = [0u32; 44];
    let mut count = 44;
    let status = query(&mut words, &mut count);
    if status != 0 || count < 38 {
        return None;
    }
    let current = u64::from(words[36]) | (u64::from(words[37]) << 32);
    let peak = (count >= 44).then(|| u64::from(words[42]) | (u64::from(words[43]) << 32));
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

pub(super) fn persist_images(docs: &Path, c: &Case, images: &[Image]) -> Result<(), String> {
    for (index, image) in images.iter().enumerate() {
        if image.width != c.edge
            || image.height != c.edge
            || image.pixels.len() != (c.edge as usize).pow(2) * 3
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
