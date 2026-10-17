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
