//! Request-owned CUDA execution policy. Dropping a failed request restores the thread state.

use super::*;
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    GenerationRequest, MemoryGeometry, MemoryRequestScope, MemoryRunOutcome,
};
use candle_gen::request_scope::{CandleRequestScopeConfig, CandleRequestScopeCore};
use std::cell::RefCell;

thread_local! {
    static ACTIVE: RefCell<Option<MemoryRunContext>> = const { RefCell::new(None) };
}

pub(crate) fn active_context() -> Option<MemoryRunContext> {
    ACTIVE.with(|active| active.borrow().clone())
}

struct RequestScope {
    core: CandleRequestScopeCore,
    active: bool,
}

impl RequestScope {
    fn clear(&mut self) {
        if self.active {
            let routes = crate::nn::attention_execution_report();
            let context = active_context();
            tracing::info!(event = "h3_attention_execution", flash_calls=routes.flash_calls,
                materialized_calls=routes.materialized_calls, maximum_query_rows=routes.maximum_query_rows,
                maximum_materialized_score_elements=routes.maximum_materialized_score_elements,
                attention_chunk_size=?context.as_ref().and_then(|c|c.selection.parameters.attention_chunk_size),
                transformer_window=?context.as_ref().and_then(|c|c.selection.parameters.transformer_window_size),
                transformer_component=?context.as_ref().and_then(|c|c.selection.parameters.transformer_window_component),
                "Observed completed H3 attention routes");
            ACTIVE.with(|active| active.borrow_mut().take());
            self.active = false;
        }
    }
}

impl Drop for RequestScope {
    fn drop(&mut self) {
        self.clear();
    }
}

impl MemoryRequestScope for RequestScope {
    fn configure_request(
        &mut self,
        request: &mut GenerationRequest,
    ) -> candle_gen::gen_core::Result<()> {
        self.core.configure_request(request)
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> candle_gen::gen_core::Result<()> {
        self.core.enter_phase(phase)
    }

    fn leave_phase(&mut self, phase: MemoryPhase) -> candle_gen::gen_core::Result<()> {
        self.core.leave_phase(phase)
    }

    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> candle_gen::gen_core::Result<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }

    fn configure_attention(&mut self, size: u32) -> candle_gen::gen_core::Result<()> {
        self.core.configure_attention(size)
    }

    fn materialize_transformer_window(
        &mut self,
        first: u32,
        count: u32,
    ) -> candle_gen::gen_core::Result<()> {
        self.core.materialize_transformer_window(first, count)
    }

    fn finish(&mut self, outcome: MemoryRunOutcome) -> candle_gen::gen_core::Result<()> {
        let result = self.core.finish(outcome);
        self.clear();
        result
    }
}

pub fn begin_request(
    contract: &MemoryProviderContract,
    device: Device,
    context: &MemoryRunContext,
) -> candle_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    candle_gen::request_scope::admit(super::safety_check_loaded(contract, context))?;
    if active_context().is_some() {
        return Err(candle_gen::gen_core::Error::Unsupported(
            "minimax_h3: a memory request is already active on this thread".into(),
        ));
    }
    let mut config = CandleRequestScopeConfig::new(
        MODEL_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::dit::MiniMaxH3DitConfig::default().num_layers,
        |_pid, edge, overlap| super::routes::validate_decode(edge, overlap),
    )?;
    config.default_frames = context.geometry.frames;
    config.attention_chunk_size = context.selection.parameters.attention_chunk_size;
    config.transformer_window = context.selection.parameters.transformer_window_size;
    let scope = RequestScope {
        core: CandleRequestScopeCore::new(config),
        active: true,
    };
    crate::nn::reset_attention_execution_report();
    ACTIVE.with(|active| *active.borrow_mut() = Some(context.clone()));
    Ok(Some(Box::new(scope)))
}
