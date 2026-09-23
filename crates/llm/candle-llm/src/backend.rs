//! What this Candle build can serve on this host, before any model is loaded (story sc-24139).
//!
//! A product reads [`backend_capabilities`] (through its runtime bundle's
//! `text_backend_capabilities()`) to offer backend-specific controls — the NVFP4 weight format,
//! the CUDA-graph switch — or to disable them with the reason. The NVFP4 answer is settled by the
//! **same** gate a load runs ([`Nvfp4Context::require`]), so a disabled control carries exactly the
//! refusal the load would have returned.

use std::sync::OnceLock;

use candle_core::{Device, DeviceLocation};
use candle_quant_kernels::Nvfp4Context;
use core_llm::{
    BackendCapabilities, Error as CoreError, FeatureSupport, LoadSpec, Quantize, TextLlmRegistry,
};

use crate::decode::graph::{
    REASON_CUDA_FEATURE_OFF, REASON_FLASH_ATTN_STREAM, REASON_LEGACY_STREAM, REASON_NOT_CUDA,
};
use crate::device::{select_device, CudaStreamKind, CUDA_STREAM_ENV};

/// This build's execution backend label (`candle-cuda`, `candle-metal`, `candle-cpu`).
pub fn backend_label() -> &'static str {
    if cfg!(feature = "cuda") {
        "candle-cuda"
    } else if cfg!(feature = "metal") {
        "candle-metal"
    } else {
        "candle-cpu"
    }
}

/// What this build can serve on the device a load would use: the device, its CUDA compute
/// capability, whether an NVFP4 load passes the device gate, and whether the CUDA-graph switch
/// (`LoadSpec::cuda_graphs`) can take effect — each unavailable feature with its reason. The
/// switch cannot take effect where a load could never get the stream capture needs: a
/// `flash-attn` build (always the legacy stream) or `CANDLE_LLM_CUDA_STREAM=legacy`.
///
/// Probed once per process (the device a load selects does not change while it runs) and cached;
/// the probe opens the load device the way a load does and, on CUDA, builds the NVFP4 context
/// (a cuBLASLt handle and the fused quantizer's compile-once module, which a later NVFP4 load
/// reuses). It never reads weights.
pub fn backend_capabilities() -> BackendCapabilities {
    static CAPABILITIES: OnceLock<BackendCapabilities> = OnceLock::new();
    CAPABILITIES
        .get_or_init(|| match select_device() {
            Ok(device) => capabilities_for_device(backend_label(), &device),
            Err(error) => no_device(backend_label(), &error.to_string()),
        })
        .clone()
}

/// The capabilities of `device` for `backend` (the probe behind [`backend_capabilities`], split
/// out so the non-CUDA answers are testable on any host).
pub fn capabilities_for_device(backend: &str, device: &Device) -> BackendCapabilities {
    let nvfp4 = match Nvfp4Context::require(device) {
        Ok(_) => FeatureSupport::available(),
        Err(refusal) => FeatureSupport::unavailable(refusal.to_string()),
    };
    let (label, compute_capability) = match device.location() {
        DeviceLocation::Cpu => ("cpu".to_string(), None),
        DeviceLocation::Metal { gpu_id } => (format!("metal:{gpu_id}"), None),
        DeviceLocation::Cuda { gpu_id } => (format!("cuda:{gpu_id}"), cuda_compute_cap(device)),
    };
    let cuda_graphs = if device.is_cuda() {
        cuda_graphs_on_cuda(
            cfg!(feature = "flash-attn"),
            std::env::var(CUDA_STREAM_ENV).ok().as_deref(),
        )
    } else {
        // The same reasons the graph runner names when it refuses a step.
        let reason = if cfg!(feature = "cuda") {
            REASON_NOT_CUDA
        } else {
            REASON_CUDA_FEATURE_OFF
        };
        FeatureSupport::unavailable(format!(
            "cuda_graphs: {reason}: CUDA graphs need a CUDA load device; this runtime loads on \
             {label}"
        ))
    };
    BackendCapabilities {
        backend: backend.to_string(),
        device: label,
        compute_capability,
        nvfp4,
        cuda_graphs,
    }
}

/// Whether an NVFP4 load of the snapshot at `spec.source` can pass every gate the load runs before
/// it reads a weight (sc-24139): the per-snapshot answer a product needs before it downloads or
/// registers a checkpoint as NVFP4, where [`backend_capabilities`] only answers for the host.
///
/// The provider is the one `registry` would load `spec` with
/// ([`TextLlmRegistry::select_for_model`] — the runtime bundle's own composition), and the answer
/// comes from that provider's own load gates, never a copy of them: the llama family's
/// [`nvfp4_model_gate`](crate::provider) (GGUF, Prism and architecture refusals) followed by the
/// device gate ([`Nvfp4Context::require`], cached in [`backend_capabilities`]); LLaVA's and both
/// StarVector providers' quantization gates, which refuse NVFP4 by name. So when a provider starts
/// serving NVFP4 for more checkpoints, this answer follows. Reads only `config.json` (and a GGUF
/// header); `spec.quantize` is ignored — the question is always NVFP4.
pub fn nvfp4_support(registry: &TextLlmRegistry, spec: &LoadSpec) -> FeatureSupport {
    nvfp4_support_with(registry, spec, backend_capabilities().nvfp4)
}

/// [`nvfp4_support`] with the device gate's answer injected, so each provider's model gate is
/// testable on a host without an sm_120 GPU.
fn nvfp4_support_with(
    registry: &TextLlmRegistry,
    spec: &LoadSpec,
    device: FeatureSupport,
) -> FeatureSupport {
    let spec = LoadSpec {
        quantize: Some(Quantize::Nvfp4),
        ..spec.clone()
    };
    let registration = match registry.select_for_model(&spec) {
        Ok(registration) => registration,
        Err(error) => {
            return FeatureSupport::unavailable(format!(
                "nvfp4: no linked provider serves this snapshot: {}",
                refusal_text(error)
            ))
        }
    };
    let provider = (registration.descriptor)().id;
    let model_gate = match provider.as_str() {
        crate::provider::PROVIDER_ID => crate::provider::nvfp4_model_gate(&spec),
        crate::llava::PROVIDER_ID => crate::llava::requested_quantization(&spec).map(|_| ()),
        crate::starvector::PROVIDER_ID => crate::starvector::nvfp4_gate(&spec),
        crate::starvector_8b::PROVIDER_ID => crate::starvector_8b::quantize_gate(&spec),
        other => Err(CoreError::Unsupported(format!(
            "nvfp4: `{other}` is not a Candle provider, so this backend cannot vouch for NVFP4 on it"
        ))),
    };
    match model_gate {
        Ok(()) => device,
        Err(error) => FeatureSupport::unavailable(refusal_text(error)),
    }
}

/// A load refusal's own words: the message of a typed refusal without the error-kind prefix, so a
/// product shows exactly what the load would say (`nvfp4: …`).
fn refusal_text(error: CoreError) -> String {
    match error {
        CoreError::Unsupported(message) | CoreError::Load(message) => message,
        other => other.to_string(),
    }
}

/// The CUDA-graph switch on a CUDA device: available unless the build or the stream switch pins
/// every load to the legacy stream (the runner's own reasons, `flash_attn_stream` /
/// `legacy_stream`).
fn cuda_graphs_on_cuda(flash_attn: bool, stream_env: Option<&str>) -> FeatureSupport {
    if flash_attn {
        return FeatureSupport::unavailable(format!(
            "cuda_graphs: {REASON_FLASH_ATTN_STREAM}: this flash-attn build keeps every model on \
             the legacy CUDA stream, which stream capture cannot record"
        ));
    }
    if matches!(
        CudaStreamKind::parse(stream_env),
        Ok(Some(CudaStreamKind::Legacy))
    ) {
        return FeatureSupport::unavailable(format!(
            "cuda_graphs: {REASON_LEGACY_STREAM}: {CUDA_STREAM_ENV}=legacy pins every model to \
             the legacy CUDA stream, which stream capture cannot record"
        ));
    }
    FeatureSupport::available()
}

/// The answer when no load device could be opened: every device feature is unavailable with the
/// error a load would hit first.
fn no_device(backend: &str, error: &str) -> BackendCapabilities {
    BackendCapabilities {
        backend: backend.to_string(),
        device: format!("unavailable ({error})"),
        compute_capability: None,
        nvfp4: FeatureSupport::unavailable(format!("nvfp4: no load device: {error}")),
        cuda_graphs: FeatureSupport::unavailable(format!("cuda_graphs: no load device: {error}")),
    }
}

#[cfg(feature = "cuda")]
fn cuda_compute_cap(device: &Device) -> Option<(u32, u32)> {
    let Device::Cuda(cuda) = device else {
        return None;
    };
    let (major, minor) = candle_quant_kernels::device_compute_cap(cuda).ok()?;
    Some((u32::try_from(major).ok()?, u32::try_from(minor).ok()?))
}

#[cfg(not(feature = "cuda"))]
fn cuda_compute_cap(_device: &Device) -> Option<(u32, u32)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_device_refuses_nvfp4_and_graphs_with_the_load_gates_reasons() {
        let caps = capabilities_for_device(backend_label(), &Device::Cpu);
        assert_eq!(caps.backend, backend_label());
        assert_eq!(caps.device, "cpu");
        assert_eq!(caps.compute_capability, None);
        assert!(!caps.nvfp4.supported);
        let nvfp4 = caps.nvfp4.reason.as_deref().unwrap();
        // The load gate's own refusal text: it names the capability and the sm_120 floor.
        assert!(nvfp4.starts_with("nvfp4: "), "{nvfp4}");
        assert!(nvfp4.contains("sm_120"), "{nvfp4}");
        assert!(!caps.cuda_graphs.supported);
        let graphs = caps.cuda_graphs.reason.as_deref().unwrap();
        assert!(graphs.starts_with("cuda_graphs: "), "{graphs}");
        let expected = if cfg!(feature = "cuda") {
            REASON_NOT_CUDA
        } else {
            REASON_CUDA_FEATURE_OFF
        };
        assert!(graphs.contains(expected), "{graphs}");
    }

    #[test]
    fn graphs_on_cuda_follow_the_stream_the_load_could_get() {
        assert!(cuda_graphs_on_cuda(false, None).supported);
        assert!(cuda_graphs_on_cuda(false, Some("own")).supported);
        let legacy = cuda_graphs_on_cuda(false, Some(" LEGACY "));
        assert!(!legacy.supported);
        let reason = legacy.reason.unwrap();
        assert!(reason.contains(REASON_LEGACY_STREAM), "{reason}");
        // One sentence: a line continuation that lost its `\` leaves a run of spaces.
        assert!(!reason.contains("  "), "{reason:?}");
        let flash = cuda_graphs_on_cuda(true, None);
        assert!(!flash.supported);
        let reason = flash.reason.unwrap();
        assert!(reason.contains(REASON_FLASH_ATTN_STREAM), "{reason}");
        assert!(!reason.contains("  "), "{reason:?}");
    }

    #[test]
    fn a_missing_device_names_the_error_in_every_refusal() {
        let caps = no_device("candle-cuda", "CUDA driver not found");
        assert!(caps.device.contains("CUDA driver not found"));
        assert!(!caps.nvfp4.supported && !caps.cuda_graphs.supported);
        assert!(caps
            .nvfp4
            .reason
            .as_deref()
            .unwrap()
            .contains("CUDA driver not found"));
    }

    fn snapshot(config: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        dir
    }

    fn source(dir: &tempfile::TempDir) -> String {
        dir.path().to_string_lossy().into_owned()
    }

    fn llama() -> serde_json::Value {
        serde_json::json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    }

    fn qwen35() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5"
        })
    }

    fn llava() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
            "text_config": {"model_type": "llama"},
            "vision_config": {"model_type": "siglip_vision_model"}
        })
    }

    fn starvector_1b() -> serde_json::Value {
        serde_json::json!({
            "model_type": "starvector",
            "starcoder_model_name": "bigcode/starcoderbase-1b",
            "image_encoder_type": "clip",
            "image_size": 224,
            "hidden_size": 2048,
            "vocab_size": 49156,
            "max_position_embeddings": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "multi_query": true
        })
    }

    fn starvector_8b() -> serde_json::Value {
        serde_json::json!({
            "model_type": "starvector",
            "starcoder_model_name": "bigcode/starcoder2-7b",
            "image_encoder_type": "siglip_384",
            "adapter_norm": "layer_norm",
            "image_size": 384,
            "hidden_size": 4608,
            "num_attention_heads": 36,
            "num_hidden_layers": 32,
            "num_kv_heads": 4,
            "vocab_size": 49152
        })
    }

    /// sc-24139: the per-snapshot NVFP4 answer comes from the gates of the provider a load would
    /// reach. With the device gate passing (an sm_120 host), only a qwen3_5 snapshot is offered;
    /// every other family is refused by name, and the request's own format is irrelevant.
    #[test]
    fn nvfp4_support_follows_the_selected_providers_own_gates() {
        let registry = crate::cuda_text_registry().unwrap();
        let sm120 = FeatureSupport::available();
        let sm89 = FeatureSupport::unavailable("nvfp4: needs sm_120; this device is sm_89");

        let qwen = snapshot(qwen35());
        let spec = LoadSpec::dense(source(&qwen));
        // The model gate passes, so the device gate answers — whichever way it answers.
        assert_eq!(nvfp4_support_with(&registry, &spec, sm120.clone()), sm120);
        assert_eq!(nvfp4_support_with(&registry, &spec, sm89.clone()), sm89);
        // The question is always NVFP4, whatever format the caller's spec carries.
        let q4 = LoadSpec {
            quantize: Some(Quantize::Q4),
            ..spec.clone()
        };
        assert_eq!(nvfp4_support_with(&registry, &q4, sm89.clone()), sm89);

        for (config, names) in [
            (llama(), "Llama"),
            (llava(), "LLaVA"),
            (starvector_1b(), "StarVector-1B"),
            (starvector_8b(), "StarVector-8B"),
        ] {
            let dir = snapshot(config);
            let answer =
                nvfp4_support_with(&registry, &LoadSpec::dense(source(&dir)), sm120.clone());
            assert!(!answer.supported, "{names}: {answer:?}");
            let reason = answer.reason.unwrap();
            assert!(reason.starts_with("nvfp4: "), "{reason}");
            assert!(reason.contains(names), "{names}: {reason}");
        }

        let prism = snapshot(serde_json::json!({"model_type": "prism_hadamard_qwen35"}));
        let answer = nvfp4_support_with(&registry, &LoadSpec::dense(source(&prism)), sm120.clone());
        assert!(!answer.supported);
        assert!(answer.reason.unwrap().contains("Prism"));

        // No linked provider serves the snapshot: refused, naming why.
        let unknown = snapshot(serde_json::json!({"model_type": "not_a_model"}));
        let answer = nvfp4_support_with(&registry, &LoadSpec::dense(source(&unknown)), sm120);
        assert!(!answer.supported);
        assert!(
            answer
                .reason
                .as_deref()
                .unwrap()
                .starts_with("nvfp4: no linked provider serves this snapshot: "),
            "{answer:?}"
        );
    }

    /// The per-snapshot answer IS the load's answer, on this host's real device gate: where the
    /// probe refuses, an NVFP4 load of the same snapshot fails with exactly the probe's reason
    /// before it reads a weight (none exist here); where the probe offers NVFP4, the load gets
    /// past every NVFP4 gate and fails only for the missing weights.
    #[test]
    fn nvfp4_support_answers_what_the_load_does() {
        let registry = crate::cuda_text_registry().unwrap();
        for config in [llama(), qwen35(), llava(), starvector_1b(), starvector_8b()] {
            let dir = snapshot(config);
            let spec = LoadSpec {
                quantize: Some(Quantize::Nvfp4),
                ..LoadSpec::dense(source(&dir))
            };
            let probe = nvfp4_support(&registry, &spec);
            let load = match registry.load_for_model(&spec) {
                Ok(_) => panic!("an NVFP4 load of {:?} succeeded", spec.source),
                Err(error) => refusal_text(error),
            };
            if probe.supported {
                assert!(!load.starts_with("nvfp4: "), "{load}");
            } else {
                assert_eq!(probe.reason.as_deref(), Some(load.as_str()));
            }
        }
    }

    #[test]
    fn the_backend_label_follows_the_build() {
        let label = backend_label();
        if cfg!(feature = "cuda") {
            assert_eq!(label, "candle-cuda");
        } else if cfg!(feature = "metal") {
            assert_eq!(label, "candle-metal");
        } else {
            assert_eq!(label, "candle-cpu");
        }
    }

    /// A CUDA build either reports a CUDA load device, or — no device could be opened — refuses
    /// both device features with a reason; never a silent skip. `REQUIRE_CUDA=1` (a GPU lane)
    /// makes a missing device a failure.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_build_reports_a_cuda_device_or_refuses_both_features_with_reasons() {
        let caps = backend_capabilities();
        assert_eq!(caps.backend, "candle-cuda");
        if !caps.device.starts_with("cuda:") {
            assert!(
                std::env::var("REQUIRE_CUDA").as_deref() != Ok("1"),
                "REQUIRE_CUDA=1 but no CUDA load device: {}",
                caps.device
            );
            for feature in [&caps.nvfp4, &caps.cuda_graphs] {
                assert!(!feature.supported, "{feature:?}");
                assert!(
                    feature.reason.as_deref().is_some_and(|r| !r.is_empty()),
                    "{feature:?}"
                );
            }
            return;
        }
        assert!(caps.compute_capability.is_some(), "{caps:?}");
    }

    /// On an sm_120 device the probe offers NVFP4 and graphs and reports the capability; below it
    /// NVFP4 carries the gate's refusal. Needs a CUDA device: without one it fails when
    /// `REQUIRE_CUDA=1` and otherwise defers to the test above, which checks the refusal.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_device_reports_its_compute_capability_and_the_nvfp4_gate() {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(error) => {
                assert!(
                    std::env::var("REQUIRE_CUDA").as_deref() != Ok("1"),
                    "REQUIRE_CUDA=1 but no CUDA device: {error}"
                );
                return;
            }
        };
        let caps = capabilities_for_device("candle-cuda", &device);
        assert_eq!(caps.device, "cuda:0");
        assert_eq!(
            caps.cuda_graphs.supported,
            !cfg!(feature = "flash-attn")
                && !std::env::var(CUDA_STREAM_ENV)
                    .is_ok_and(|v| v.trim().eq_ignore_ascii_case("legacy")),
            "{:?}",
            caps.cuda_graphs
        );
        let cap = caps
            .compute_capability
            .expect("a CUDA device reports its capability");
        if cap >= (12, 0) {
            assert!(caps.nvfp4.supported, "{:?}", caps.nvfp4);
        } else {
            assert!(!caps.nvfp4.supported);
            assert!(caps.nvfp4.reason.as_deref().unwrap().contains("sm_120"));
        }
    }
}
