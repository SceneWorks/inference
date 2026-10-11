//! Benchmark-only synchronized timings. Compare whole-forward enabled/disabled arms.
use candle_gen::candle_core::Device;
use candle_gen::Result;
use std::{cell::RefCell, time::Instant};

#[derive(Clone, Default, Debug)]
pub struct Sample {
    pub seconds: [f64; 4],
    pub calls: [usize; 4],
}
thread_local! {
    static STATE: RefCell<(bool, Sample)> = RefCell::new((false, Sample::default()));
}
pub struct Scope;
impl Scope {
    pub fn new(enabled: bool) -> Self {
        STATE.with(|s| *s.borrow_mut() = (enabled, Sample::default()));
        Self
    }
    pub fn finish(self) -> Sample {
        STATE.with(|s| std::mem::take(&mut s.borrow_mut().1))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        STATE.with(|s| s.borrow_mut().0 = false);
    }
}
pub(super) fn start(device: &Device) -> Result<Option<Instant>> {
    if !STATE.with(|s| s.borrow().0) {
        return Ok(None);
    }
    device.synchronize()?;
    Ok(Some(Instant::now()))
}
pub(super) fn stop(start: Option<Instant>, device: &Device, phase: usize) -> Result<()> {
    if let Some(start) = start {
        device.synchronize()?;
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.1.seconds[phase] += start.elapsed().as_secs_f64();
            s.1.calls[phase] += 1;
        });
    }
    Ok(())
}
