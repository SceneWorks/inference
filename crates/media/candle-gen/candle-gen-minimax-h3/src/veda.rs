//! Job-local attention controls. CUDA sparse execution replaces this dense fallback at integration.
use std::{cell::RefCell, rc::Rc};

use candle_gen::{
    candle_core::Tensor,
    gen_core::{VedaExecutionReport, VedaRequest},
    Result,
};

pub const COMPONENT: &str = "veda_predictor";

pub type ReportSink = Rc<RefCell<VedaExecutionReport>>;

pub struct VedaRuntime {
    request: VedaRequest,
    report: ReportSink,
    predictor_missing: bool,
    #[cfg(test)]
    pub(crate) calls: RefCell<Vec<AttentionCall>>,
}
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AttentionCall {
    pub layer: usize,
    pub step: usize,
    pub full_attention: bool,
    pub request: VedaRequest,
    pub shape: Vec<usize>,
}
impl VedaRuntime {
    pub fn prepare(
        request: &VedaRequest,
        predictor: Option<&std::path::Path>,
        report: ReportSink,
    ) -> Option<Self> {
        if !request.enabled {
            return None;
        }
        Some(Self {
            request: request.clone(),
            report,
            predictor_missing: predictor.is_none(),
            #[cfg(test)]
            calls: Default::default(),
        })
    }
    pub fn attention(
        &self,
        layer: usize,
        step: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        #[cfg(test)]
        self.calls.borrow_mut().push(AttentionCall {
            layer,
            step,
            full_attention: self.request.full_attention(layer, step),
            request: self.request.clone(),
            shape: q.dims().to_vec(),
        });
        // Keep intentional full attention distinguishable from unavailable sparse execution.
        let intentional = self.request.full_attention(layer, step);
        if !intentional {
            self.report.borrow_mut().fallback_reason = Some(if self.predictor_missing {
                "Veda predictor component is missing; repair/install the managed component; dense attention executed"
            } else {
                "Veda sparse kernel unavailable on this Candle device; dense attention executed"
            }.into());
        }
        let out = crate::nn::sdpa(q, k, v, scale)?;
        let (batch, heads, rows, _) = q.dims4()?;
        let pairs = batch as u64 * heads as u64 * rows as u64 * rows as u64;
        let mut report = self.report.borrow_mut();
        report.dense_calls += 1;
        report.selected_token_pairs += pairs;
        report.dense_token_pairs += pairs;
        Ok(out)
    }
}
