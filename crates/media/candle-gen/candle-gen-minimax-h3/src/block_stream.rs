//! Request-owned bounded CUDA block residency. Frozen runtime-format blocks stay on the host;
//! windows copy those bytes and their adapter factors without repeating checkpoint conversion.

use std::cell::RefCell;

use candle_gen::gen_core::{CancelFlag, GenerationRequest, LoadShape, TransformerComponent};
use candle_gen::{CandleError, Result};

#[derive(Clone)]
pub(crate) struct Policy {
    pub window: usize,
    pub cancel: CancelFlag,
}

thread_local! {
    static ACTIVE: RefCell<Option<Policy>> = const { RefCell::new(None) };
}

pub(crate) fn current() -> Option<Policy> {
    ACTIVE.with(|active| active.borrow().clone())
}

pub(crate) struct RequestGuard {
    device: candle_gen::candle_core::Device,
}

impl RequestGuard {
    pub(crate) fn enter(
        request: &GenerationRequest,
        shape: LoadShape,
        device: &candle_gen::candle_core::Device,
    ) -> Result<Self> {
        let policy = request
            .memory
            .filter(|m| m.stream_transformer_blocks)
            .map(|memory| {
                if shape != LoadShape::DeferredMaterialization {
                    return Err(CandleError::Msg(
                        "minimax-h3 streaming requires a deferred-materialization load".into(),
                    ));
                }
                if memory.transformer_window_component != Some(TransformerComponent::Both) {
                    return Err(CandleError::Msg(
                        "minimax-h3 streaming requires both text encoder and DiT components".into(),
                    ));
                }
                let window = memory.transformer_window_size.unwrap_or(1) as usize;
                if window != 1 {
                    return Err(CandleError::Msg(
                        "minimax-h3 supports a one-block transformer window".into(),
                    ));
                }
                Ok(Policy {
                    window,
                    cancel: request.cancel.clone(),
                })
            })
            .transpose()?;
        candle_gen::execution::policy(
            "h3_transformer_residency",
            serde_json::json!({
                "block_window": policy.as_ref().map(|p| p.window),
                "load_shape": format!("{shape:?}"), "component_offload": "sequential"
            }),
        );
        ACTIVE.with(|active| {
            if active.borrow().is_some() {
                return Err(CandleError::Msg(
                    "nested minimax-h3 streaming request".into(),
                ));
            }
            *active.borrow_mut() = policy;
            Ok(Self {
                device: device.clone(),
            })
        })
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.borrow_mut().take());
        let _ = self.device.synchronize();
    }
}

/// Check source representation before any tensor is materialized. Native specialized bases
/// currently refuse cross-device migration; their valid resident route remains available.
pub(crate) fn component_supported(dir: &std::path::Path) -> Result<bool> {
    let files = candle_gen::loader::sorted_safetensors(dir, "minimax-h3 streaming source")?;
    for file in files {
        let headers = candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file)
            .map_err(|error| CandleError::Msg(error.to_string()))?;
        if headers.iter().any(|header| {
            [
                ".comfy_quant",
                ".weight_codebook",
                ".weight_s_rel",
                ".weight_s_channel",
                ".weight_scale",
            ]
            .iter()
            .any(|suffix| header.name.ends_with(suffix))
        }) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::GenerationMemory;

    pub(crate) fn request() -> GenerationRequest {
        GenerationRequest {
            memory: Some(GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn refused_and_failed_request_scopes_do_not_poison_the_next_request() {
        let request = request();
        let device = candle_gen::candle_core::Device::Cpu;
        assert!(RequestGuard::enter(&request, LoadShape::EagerMaterialization, &device).is_err());
        assert!(current().is_none());
        {
            let _guard =
                RequestGuard::enter(&request, LoadShape::DeferredMaterialization, &device).unwrap();
            assert_eq!(current().unwrap().window, 1);
            assert!(
                RequestGuard::enter(&request, LoadShape::DeferredMaterialization, &device).is_err()
            );
            assert!(current().is_some());
        }
        assert!(current().is_none());
        let _next =
            RequestGuard::enter(&request, LoadShape::DeferredMaterialization, &device).unwrap();
        assert!(current().is_some());
    }

    #[test]
    fn component_and_window_domain_are_enforced_before_loading() {
        let device = candle_gen::candle_core::Device::Cpu;
        for (component, window) in [
            (TransformerComponent::Dit, 1),
            (TransformerComponent::Both, 0),
            (TransformerComponent::Both, 2),
        ] {
            let mut request = request();
            let memory = request.memory.as_mut().unwrap();
            memory.transformer_window_component = Some(component);
            memory.transformer_window_size = Some(window);
            assert!(
                RequestGuard::enter(&request, LoadShape::DeferredMaterialization, &device).is_err()
            );
            assert!(current().is_none());
        }
    }
}
