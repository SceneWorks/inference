//! The warm-set decision shared by the two resident Candle generators, MiniMax-H3 video
//! ([`crate::model`]) and MiniMax-H3-Image ([`crate::image_model`]).
//!
//! A **resident** request (no request memory, or one that neither stages nor streams) may keep the
//! generator's components on the device between requests instead of re-reading them from disk.
//! It does so only when the provider's header-derived warm peak, plus [`WARM_RESERVE_BYTES`], fits
//! the device's free memory plus what the generator already holds warm, and fits an admitted
//! context's budget. Any other request releases the warm components and stages as before.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::GenerationRequest;

/// Device memory a warm request leaves free above its estimate — the same 2 GiB allocator slack
/// the worker's admission keeps.
pub(crate) const WARM_RESERVE_BYTES: u64 = 2 << 30;

/// Whether a request may keep its components warm: no request memory (a resident admission
/// carries none) or one that neither stages nor streams.
pub(crate) fn keeps_components_warm(req: &GenerationRequest) -> bool {
    req.memory
        .is_none_or(|memory| !memory.stage_residency && !memory.stream_transformer_blocks)
}

/// Whether a warm set needing `need` device bytes at its peak fits, given the device's `free`
/// bytes (`None` off CUDA, where nothing is kept warm), the `held` bytes this generator already
/// keeps warm (they are part of `need` and already allocated) and, when the request carries an
/// admitted context, that context's `admitted` budget (which credits this generator's held warm
/// set as reclaimable), so a warm set never outgrows what the caller granted.
pub(crate) fn warm_set_fits(
    need: u64,
    held: u64,
    free: Option<u64>,
    admitted: Option<u64>,
) -> bool {
    free.is_some_and(|free| need.saturating_add(WARM_RESERVE_BYTES) <= free.saturating_add(held))
        && admitted.is_none_or(|budget| need <= budget)
}

/// The device's free bytes as the CUDA driver reports them after pending work completes; `None`
/// on any other device.
pub(crate) fn device_free_bytes(device: &Device) -> Option<u64> {
    #[cfg(feature = "cuda")]
    if let Ok(cuda) = device.as_cuda_device() {
        device.synchronize().ok()?;
        return cuda
            .cuda_stream()
            .context()
            .mem_get_info()
            .ok()
            .map(|(free, _total)| free as u64);
    }
    let _ = device;
    None
}

/// One warm-set decision and the figures it used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WarmDecision {
    pub(crate) keep_warm: bool,
    /// The warm estimate's device peak.
    pub(crate) need: u64,
    /// Bytes the generator already held warm when it decided.
    pub(crate) held: u64,
    pub(crate) free: Option<u64>,
    pub(crate) admitted: Option<u64>,
}

impl WarmDecision {
    /// Decide against the device's free memory now and the active request's admitted budget.
    pub(crate) fn decide(need: u64, held: u64, device: &Device) -> Self {
        let free = device_free_bytes(device);
        let admitted = crate::memory_strategy::active_context()
            .map(|context| context.budget.effective_bytes());
        Self {
            keep_warm: warm_set_fits(need, held, free, admitted),
            need,
            held,
            free,
            admitted,
        }
    }

    /// Log the decision as `event` (`minimax_h3_warm_decision`,
    /// `minimax_h3_image_warm_decision`). `components` names what is kept warm.
    pub(crate) fn log(&self, event: &'static str, provider: &'static str, components: &str) {
        use crate::image_memory_strategy::admission::decimal_gb;
        tracing::info!(
            event,
            keep_warm = self.keep_warm,
            warm_peak_gb = decimal_gb(self.need),
            held_warm_gb = decimal_gb(self.held),
            device_free_gb = self.free.map(decimal_gb),
            admitted_budget_gb = self.admitted.map(decimal_gb),
            reserve_gb = decimal_gb(WARM_RESERVE_BYTES),
            "{provider}: {}",
            if self.keep_warm {
                format!("the warm set fits; keeping the {components} resident")
            } else {
                "the warm set does not fit; staging this request".to_owned()
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only a request that neither stages nor streams may keep the components warm; a resident
    /// admission carries no request memory at all.
    #[test]
    fn only_a_resident_request_keeps_components_warm() {
        use candle_gen::gen_core::GenerationMemory;
        let mut req = GenerationRequest::default();
        assert!(
            keeps_components_warm(&req),
            "no request memory is a resident request"
        );
        req.memory = Some(GenerationMemory::default());
        assert!(keeps_components_warm(&req));
        req.memory = Some(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        });
        assert!(!keeps_components_warm(&req), "a staged request stages");
        req.memory = Some(GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: true,
            ..Default::default()
        });
        assert!(!keeps_components_warm(&req), "a streamed request streams");
    }

    /// The warm set fits when its peak plus the reserve is within the free memory plus what is
    /// already held warm, and within an admitted context's budget; with no CUDA free figure
    /// nothing is kept warm.
    #[test]
    fn the_warm_set_fits_only_within_free_plus_held_memory() {
        const GB: u64 = 1_000_000_000;
        // q8-sized warm set on an empty 96 GB-class card: fits.
        assert!(warm_set_fits(66 * GB, 0, Some(100 * GB), None));
        // The same set already warm: the card reports only what is left, plus what is held.
        assert!(warm_set_fits(66 * GB, 60 * GB, Some(40 * GB), None));
        // A ~126 GB warm set never fits that card, warm or cold.
        assert!(!warm_set_fits(126 * GB, 0, Some(100 * GB), None));
        // The reserve is kept: an exact fit with no slack is refused.
        assert!(!warm_set_fits(100 * GB, 0, Some(100 * GB), None));
        assert!(warm_set_fits(
            100 * GB - WARM_RESERVE_BYTES,
            0,
            Some(100 * GB),
            None
        ));
        // An admitted context's budget bounds the warm set even when the card has room.
        assert!(warm_set_fits(66 * GB, 0, Some(100 * GB), Some(66 * GB)));
        assert!(!warm_set_fits(66 * GB, 0, Some(100 * GB), Some(60 * GB)));
        // Off CUDA there is no free figure, so nothing is kept warm.
        assert!(!warm_set_fits(GB, 0, None, None));
        assert_eq!(device_free_bytes(&Device::Cpu), None);
    }
}
