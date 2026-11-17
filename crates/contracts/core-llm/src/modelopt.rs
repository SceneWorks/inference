//! NVIDIA **ModelOpt NVFP4** checkpoints (SW-7099): the tensor-free half both LLM backends share —
//! recognising the format from `config.json`, and naming the companion tensors it stores beside
//! each quantized projection.
//!
//! A ModelOpt NVFP4 export (`quantization_config.quant_method == "modelopt"`, `quant_algo ==
//! "NVFP4"`, group 16) stores every quantized linear `<stem>` as
//!
//! | tensor | dtype | shape | meaning |
//! |---|---|---|---|
//! | `<stem>.weight` | U8 | `[out, in / 2]` | E2M1 codes, even column in the low nibble |
//! | `<stem>.weight_scale` | F8_E4M3 | `[out, in / 16]` | per-16-block scale, linear row-major |
//! | `<stem>.weight_scale_2` | F32 | `[]` | per-tensor dequant multiplier |
//! | `<stem>.input_scale` | F32 | `[]` | calibrated activation scale (static W4A4) |
//!
//! Every other tensor (embeddings, norms, the excluded projections, a vision tower) is stored
//! dense. Which projections are quantized is read from the tensors themselves — a `.weight` with a
//! `.weight_scale` sibling — never guessed from a module list.
//!
//! Only CUDA Blackwell (sm_120) executes these weights natively (Candle). Every other runtime
//! loads a tier converted from them (bf16, then Q8 / Q4); [`dense_runtime_refusal`] is the one
//! message such a runtime gives when it is pointed at the NVFP4 snapshot itself, so it is never
//! read as dense U8 weights.

use serde_json::Value;

/// Suffix of a quantized projection's per-block scale (F8_E4M3).
pub const WEIGHT_SCALE: &str = ".weight_scale";
/// Suffix of a quantized projection's per-tensor dequant multiplier (F32 scalar).
pub const WEIGHT_SCALE_2: &str = ".weight_scale_2";
/// Suffix of a quantized projection's calibrated activation scale (F32 scalar).
pub const INPUT_SCALE: &str = ".input_scale";
/// The block size ModelOpt NVFP4 uses (and the only one this runtime reads).
pub const GROUP_SIZE: usize = 16;

/// A recognised ModelOpt NVFP4 checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelOptNvfp4 {
    /// Elements per E4M3 block scale (always [`GROUP_SIZE`]).
    pub group_size: usize,
}

/// The `quantization_config` block of `config` (top level, else `text_config`).
fn quantization_config(config: &Value) -> Option<&Value> {
    config.get("quantization_config").or_else(|| {
        config
            .get("text_config")
            .and_then(|text| text.get("quantization_config"))
    })
}

/// Whether `config` declares a ModelOpt checkpoint, and which.
///
/// `Ok(None)`: not a ModelOpt checkpoint (no block, or another `quant_method`). `Ok(Some)`: NVFP4
/// at group 16. `Err`: a ModelOpt checkpoint this runtime does not read (another algorithm or
/// group size) — refused by name rather than read as something else.
pub fn modelopt_nvfp4(config: &Value) -> Result<Option<ModelOptNvfp4>, String> {
    let Some(block) = quantization_config(config) else {
        return Ok(None);
    };
    if block.get("quant_method").and_then(Value::as_str) != Some("modelopt") {
        return Ok(None);
    }
    let algo = block.get("quant_algo").and_then(Value::as_str);
    if algo != Some("NVFP4") {
        return Err(format!(
            "ModelOpt quant_algo {algo:?} is not supported; only NVFP4 checkpoints are read"
        ));
    }
    // The top-level `group_size`, else the weights entry of the first config group.
    let group = block.get("group_size").and_then(Value::as_u64).or_else(|| {
        block
            .get("config_groups")
            .and_then(Value::as_object)
            .and_then(|groups| groups.values().next())
            .and_then(|g| g.get("weights"))
            .and_then(|w| w.get("group_size"))
            .and_then(Value::as_u64)
    });
    if group != Some(GROUP_SIZE as u64) {
        return Err(format!(
            "ModelOpt NVFP4 group_size {group:?} is not supported; NVFP4 blocks are {GROUP_SIZE}"
        ));
    }
    Ok(Some(ModelOptNvfp4 {
        group_size: GROUP_SIZE,
    }))
}

/// Whether `key` is one of the ModelOpt companion tensors stored beside a quantized projection
/// (`.weight_scale`, `.weight_scale_2`, `.input_scale`) rather than a weight of its own.
pub fn is_companion(key: &str) -> bool {
    key.ends_with(WEIGHT_SCALE) || key.ends_with(WEIGHT_SCALE_2) || key.ends_with(INPUT_SCALE)
}

/// The projection stem of a quantized `<stem>.weight` whose `<stem>.weight_scale` sibling
/// `contains` reports present, else `None` (a dense weight).
pub fn quantized_stem(key: &str, contains: impl Fn(&str) -> bool) -> Option<&str> {
    let stem = key.strip_suffix(".weight")?;
    contains(&format!("{stem}{WEIGHT_SCALE}")).then_some(stem)
}

/// The refusal a runtime that cannot execute NVFP4 gives for a ModelOpt NVFP4 snapshot: the
/// stored U8 codes are not dense weights, and the supported route is a converted tier.
pub fn dense_runtime_refusal(runtime: &str) -> String {
    format!(
        "nvfp4: this snapshot stores ModelOpt NVFP4 projections, which {runtime} does not execute \
         natively; load a tier converted from it instead (candle-llm example \
         `modelopt_nvfp4_to_bf16`, then `prepare_snapshot` for Q8 / Q4)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn modelopt(algo: &str, group: u64) -> Value {
        json!({
            "model_type": "qwen3_5",
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": algo,
                "group_size": group,
            }
        })
    }

    #[test]
    fn a_modelopt_nvfp4_config_is_recognised_at_group_16() {
        assert_eq!(
            modelopt_nvfp4(&modelopt("NVFP4", 16)),
            Ok(Some(ModelOptNvfp4 { group_size: 16 }))
        );
    }

    #[test]
    fn the_group_size_may_live_only_in_the_config_group() {
        let config = json!({"quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {"group_0": {"weights": {"group_size": 16, "num_bits": 4}}},
        }});
        assert!(modelopt_nvfp4(&config).unwrap().is_some());
    }

    #[test]
    fn a_nested_text_config_block_is_read() {
        let config = json!({"text_config": modelopt("NVFP4", 16)});
        assert!(modelopt_nvfp4(&config).unwrap().is_some());
    }

    #[test]
    fn other_quantization_methods_are_not_modelopt() {
        assert_eq!(modelopt_nvfp4(&json!({"model_type": "qwen3_5"})), Ok(None));
        let mlx = json!({"quantization_config": {"bits": 4, "group_size": 64}});
        assert_eq!(modelopt_nvfp4(&mlx), Ok(None));
    }

    #[test]
    fn an_unsupported_modelopt_variant_is_refused_by_name() {
        let fp8 = modelopt_nvfp4(&modelopt("FP8", 16)).unwrap_err();
        assert!(fp8.contains("FP8"), "{fp8}");
        let group = modelopt_nvfp4(&modelopt("NVFP4", 32)).unwrap_err();
        assert!(group.contains("32"), "{group}");
    }

    #[test]
    fn companions_and_stems_follow_the_tensor_names() {
        let names = ["lm_head.weight", "lm_head.weight_scale", "norm.weight"];
        let contains = |k: &str| names.contains(&k);
        assert_eq!(quantized_stem("lm_head.weight", contains), Some("lm_head"));
        assert_eq!(quantized_stem("norm.weight", contains), None);
        assert_eq!(quantized_stem("lm_head.weight_scale", contains), None);
        for key in ["a.weight_scale", "a.weight_scale_2", "a.input_scale"] {
            assert!(is_companion(key), "{key}");
        }
        assert!(!is_companion("a.weight"));
    }
}
