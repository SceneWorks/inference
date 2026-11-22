//! Request-owned bounded CUDA block residency. Frozen runtime-format blocks stay on the host;
//! windows copy those bytes and their adapter factors without repeating checkpoint conversion.

use std::cell::RefCell;

use candle_gen::gen_core::{CancelFlag, GenerationRequest, LoadShape, TransformerComponent};
use candle_gen::{CandleError, Result};

#[derive(Clone)]
pub(crate) struct Policy {
    pub window: usize,
    pub device: candle_gen::candle_core::Device,
    pub cancel: CancelFlag,
}

thread_local! {
    static ACTIVE: RefCell<Option<Policy>> = const { RefCell::new(None) };
}

pub(crate) fn current() -> Option<Policy> {
    ACTIVE.with(|active| active.borrow().clone())
}

/// The request scope rung 4 runs inside: while it lives, the text encoder and the DiT read the
/// one-block window policy `request.memory` selects. The generator enters it per request; a
/// real-weights gate that drives the streamed DiT directly enters it the same way.
pub struct RequestGuard {
    device: candle_gen::candle_core::Device,
}

impl RequestGuard {
    /// Install `request`'s window policy on this thread for the guard's lifetime. A request that
    /// does not stream installs none; streaming needs a deferred load, both transformers and a
    /// one-block window, and a scope is refused while another is active on the thread.
    pub fn enter(
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
                    device: device.clone(),
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

/// Check the source representation across the exact shards opened by the window loader.
pub(crate) fn component_supported(dir: &std::path::Path) -> Result<bool> {
    use candle_gen::gen_core::weightsmeta::{self, Dtype};
    let files = candle_gen::loader::sorted_safetensors(dir, "minimax-h3 streaming source")?;
    let mut weights = std::collections::BTreeMap::new();
    for file in files {
        for h in weightsmeta::safetensors_path_tensor_headers(&file)
            .map_err(|e| CandleError::Msg(e.to_string()))?
        {
            if weights.insert(h.name.clone(), h.dtype).is_some() {
                return Err(CandleError::Msg(format!(
                    "duplicate H3 source tensor {}",
                    h.name
                )));
            }
        }
    }
    // A descriptor or scale may be in another shard. FP8 windows bind all original
    // operands; the integer-specialized H3 operators still have a resident-only contract.
    Ok(!weights.keys().any(|name| {
        if [".weight_codebook", ".weight_s_rel", ".weight_s_channel"]
            .iter()
            .any(|s| name.ends_with(s))
        {
            return true;
        }
        [".comfy_quant", ".weight_scale"]
            .iter()
            .find_map(|s| name.strip_suffix(s))
            .is_some_and(|base| {
                !matches!(
                    weights.get(format!("{base}.weight").as_str()),
                    Some(Dtype::F8_E4M3 | Dtype::F8_E5M2)
                )
            })
    }))
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
